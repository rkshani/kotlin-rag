package io.github.rkshani.rag.core.loader

import io.github.rkshani.rag.core.model.SourceDocument
import org.xml.sax.Attributes
import org.xml.sax.InputSource
import org.xml.sax.helpers.DefaultHandler
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction
import java.util.zip.ZipInputStream
import javax.xml.parsers.SAXParserFactory

/** .txt, .md and friends. */
class PlainTextLoader(
    private val extensions: Set<String> = setOf("txt", "md", "markdown", "text", "csv", "log"),
) : DocumentLoader {

    override fun supports(input: DocumentInput): Boolean =
        input.extension in extensions || input.mimeType?.startsWith("text/") == true

    override fun load(input: DocumentInput, limits: LoadLimits): SourceDocument {
        val bytes = BoundedInputStream(input.open(), limits.maxBytes, input.name).use { stream ->
            val buffer = ByteArrayOutputStream()
            stream.copyTo(buffer)
            buffer.toByteArray()
        }
        return SourceDocument(
            id = input.id,
            text = decode(bytes),
            title = input.title,
            metadata = input.metadata + ("title" to input.title),
        )
    }

    /**
     * BOM first, then strict UTF-8, then Windows-1252.
     *
     * Decoding with replacement would "succeed" on any bytes and index a file
     * full of U+FFFD. Strict decoding makes a wrong guess fail loudly so the
     * next candidate can be tried.
     */
    private fun decode(bytes: ByteArray): String {
        fun startsWith(vararg bom: Int) = bytes.size >= bom.size && bom.indices.all { bytes[it] == bom[it].toByte() }
        return when {
            startsWith(0xEF, 0xBB, 0xBF) -> String(bytes, 3, bytes.size - 3, Charsets.UTF_8)
            startsWith(0xFF, 0xFE) -> String(bytes, 2, bytes.size - 2, Charsets.UTF_16LE)
            startsWith(0xFE, 0xFF) -> String(bytes, 2, bytes.size - 2, Charsets.UTF_16BE)
            else -> strict(bytes, Charsets.UTF_8) ?: String(bytes, Charset.forName("windows-1252"))
        }
    }

    private fun strict(bytes: ByteArray, charset: Charset): String? = try {
        charset.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes))
            .toString()
    } catch (e: CharacterCodingException) {
        null
    }
}

/**
 * Word .docx, read with nothing but java.util.zip and SAX.
 *
 * A .docx is a zip archive whose body text lives in `word/document.xml`. Pulling
 * in a full Office library for that costs megabytes of dependency for what is a
 * streaming XML walk.
 */
class DocxLoader : DocumentLoader {

    override fun supports(input: DocumentInput): Boolean =
        input.extension == "docx" ||
            input.mimeType == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    override fun load(input: DocumentInput, limits: LoadLimits): SourceDocument {
        val handler = BodyHandler(limits.maxChars, input.name)
        var found = false

        ZipInputStream(BoundedInputStream(input.open(), limits.maxBytes, input.name)).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (entry.name != "word/document.xml") continue
                found = true
                // The archive bound above counts compressed bytes; this one counts
                // what actually comes out, which is what a zip bomb inflates.
                val expanded = BoundedInputStream(zip, limits.maxExpandedBytes, "${input.name} (expanded)")
                secureParser().parse(InputSource(expanded), handler)
                break
            }
        }
        if (!found) throw DocumentLoadException("'${input.name}' is not a Word document (no word/document.xml)")

        return SourceDocument(
            id = input.id,
            text = handler.text.toString(),
            title = input.title,
            pageOffsets = handler.pageOffsets(),
            metadata = input.metadata + ("title" to input.title),
        )
    }

    /**
     * External entities and DTDs are switched off. A document is untrusted input,
     * and an XML parser left on defaults will read local files or open network
     * connections when the XML asks it to (XXE). Each feature is set separately
     * because Android's parser rejects some of the names — it also never resolves
     * external entities to begin with.
     */
    private fun secureParser() = SAXParserFactory.newInstance().apply {
        isNamespaceAware = true
        listOf(
            "http://apache.org/xml/features/disallow-doctype-decl" to true,
            "http://xml.org/sax/features/external-general-entities" to false,
            "http://xml.org/sax/features/external-parameter-entities" to false,
            "http://apache.org/xml/features/nonvalidating/load-external-dtd" to false,
        ).forEach { (feature, value) -> runCatching { setFeature(feature, value) } }
        runCatching { isXIncludeAware = false }
    }.newSAXParser()

    private class BodyHandler(private val maxChars: Int, private val name: String) : DefaultHandler() {
        val text = StringBuilder()
        private val pageStarts = ArrayList<Int>()
        private var inText = false

        fun pageOffsets(): List<Int> = if (pageStarts.isEmpty()) emptyList() else listOf(0) + pageStarts

        override fun startElement(uri: String?, localName: String?, qName: String?, attributes: Attributes?) {
            when (local(localName, qName)) {
                // Only <w:t> is body text. <w:instrText> (field codes) and
                // <w:delText> (tracked deletions) are deliberately not read.
                "t" -> inText = true
                "tab" -> text.append('\t')
                "cr" -> text.append('\n')
                "br" -> if (attribute(attributes, "type") == "page") markPage() else text.append('\n')
                "lastRenderedPageBreak" -> markPage()
            }
        }

        override fun endElement(uri: String?, localName: String?, qName: String?) {
            when (local(localName, qName)) {
                "t" -> inText = false
                "p" -> text.append("\n\n")
                "tc" -> text.append('\t')
                "tr" -> text.append('\n')
            }
        }

        override fun characters(ch: CharArray, start: Int, length: Int) {
            if (!inText) return
            if (text.length + length > maxChars) {
                throw DocumentTooLargeException("$name has more than $maxChars characters of text")
            }
            text.append(ch, start, length)
        }

        /**
         * Word writes both an explicit page break and a "last rendered" marker
         * for the same boundary, so a mark with no text since the previous one
         * is the same page turning, not a new one.
         */
        private fun markPage() {
            val at = text.length
            if (at == 0 || pageStarts.lastOrNull() == at) return
            if (text.substring(pageStarts.lastOrNull() ?: 0, at).isBlank()) return
            pageStarts.add(at)
        }

        private fun local(localName: String?, qName: String?): String =
            localName?.takeIf { it.isNotEmpty() } ?: qName.orEmpty().substringAfter(':')

        private fun attribute(attributes: Attributes?, name: String): String? {
            attributes ?: return null
            for (i in 0 until attributes.length) {
                val key = attributes.getLocalName(i)?.takeIf { it.isNotEmpty() }
                    ?: attributes.getQName(i).orEmpty().substringAfter(':')
                if (key == name) return attributes.getValue(i)
            }
            return null
        }
    }
}
