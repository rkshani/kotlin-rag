package io.github.rkshani.rag.core.loader

import io.github.rkshani.rag.core.text.SearchText
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private fun input(name: String, bytes: ByteArray, size: Long? = bytes.size.toLong()) =
    DocumentInput(id = name, name = name, sizeBytes = size) { ByteArrayInputStream(bytes) }

private fun docx(documentXml: String, extra: Map<String, ByteArray> = emptyMap()): ByteArray {
    val out = ByteArrayOutputStream()
    ZipOutputStream(out).use { zip ->
        zip.putNextEntry(ZipEntry("[Content_Types].xml"))
        zip.write("<Types/>".toByteArray())
        extra.forEach { (name, bytes) ->
            zip.putNextEntry(ZipEntry(name))
            zip.write(bytes)
        }
        zip.putNextEntry(ZipEntry("word/document.xml"))
        zip.write(documentXml.toByteArray())
    }
    return out.toByteArray()
}

private const val W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

private fun body(inner: String) = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="$W"><w:body>$inner</w:body></w:document>"""

class PlainTextLoaderTest {

    private val loaders = DocumentLoaders(DocumentLoaders.defaults())

    @Test
    fun `reads utf8 including urdu`() {
        val doc = loaders.load(input("notes.txt", "Hello\nیہ اردو ہے".toByteArray(Charsets.UTF_8)))
        assertEquals("Hello\nیہ اردو ہے", doc.text)
        assertEquals("notes", doc.title)
        assertEquals("notes", doc.metadata["title"])
    }

    @Test
    fun `strips a utf8 byte order mark`() {
        val bytes = byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) + "text".toByteArray()
        assertEquals("text", loaders.load(input("a.txt", bytes)).text)
    }

    @Test
    fun `reads utf16 with a byte order mark`() {
        val bytes = byteArrayOf(0xFF.toByte(), 0xFE.toByte()) + "اردو".toByteArray(Charsets.UTF_16LE)
        assertEquals("اردو", loaders.load(input("a.txt", bytes)).text)
    }

    @Test
    fun `falls back to windows-1252 instead of producing replacement characters`() {
        val bytes = "café".toByteArray(charset("windows-1252"))
        val text = loaders.load(input("a.txt", bytes)).text
        assertEquals("café", text)
        assertFalse(text.contains('�'))
    }

    @Test
    fun `rejects a file over the byte limit before reading it`() {
        val limits = LoadLimits(maxBytes = 10)
        assertFailsWith<DocumentTooLargeException> { loaders.load(input("a.txt", ByteArray(100)), limits) }
    }

    @Test
    fun `rejects an oversized stream even when the declared size lies`() {
        val limits = LoadLimits(maxBytes = 10)
        assertFailsWith<DocumentTooLargeException> {
            loaders.load(input("a.txt", ByteArray(100) { 'a'.code.toByte() }, size = null), limits)
        }
    }

    @Test
    fun `rejects text over the character limit`() {
        val limits = LoadLimits(maxChars = 5)
        assertFailsWith<DocumentTooLargeException> { loaders.load(input("a.txt", "abcdefgh".toByteArray()), limits) }
    }

    @Test
    fun `unknown formats are refused clearly`() {
        assertFailsWith<UnsupportedDocumentException> { loaders.load(input("image.png", ByteArray(4))) }
        assertFalse(loaders.supports(input("image.png", ByteArray(4))))
    }
}

class DocxLoaderTest {

    private val loaders = DocumentLoaders(DocumentLoaders.defaults())

    @Test
    fun `extracts paragraphs tabs and line breaks`() {
        val xml = body(
            "<w:p><w:r><w:t>First</w:t></w:r><w:r><w:tab/><w:t>tabbed</w:t></w:r></w:p>" +
                "<w:p><w:r><w:t>Second</w:t><w:br/><w:t>line</w:t></w:r></w:p>",
        )
        val doc = loaders.load(input("report.docx", docx(xml)))
        assertEquals("First\ttabbed\n\nSecond\nline", doc.text.trim())
        assertEquals("report", doc.title)
    }

    @Test
    fun `ignores field codes and tracked deletions`() {
        val xml = body(
            "<w:p><w:r><w:instrText>PAGE \\* MERGEFORMAT</w:instrText></w:r>" +
                "<w:r><w:delText>deleted words</w:delText></w:r><w:r><w:t>kept</w:t></w:r></w:p>",
        )
        assertEquals("kept", loaders.load(input("a.docx", docx(xml))).text.trim())
    }

    @Test
    fun `records page breaks as page offsets`() {
        val xml = body(
            "<w:p><w:r><w:t>Page one text</w:t></w:r></w:p>" +
                "<w:p><w:r><w:br w:type=\"page\"/><w:lastRenderedPageBreak/><w:t>Page two text</w:t></w:r></w:p>",
        )
        val doc = loaders.load(input("a.docx", docx(xml)))
        assertEquals(2, doc.pageOffsets.size, "both markers describe the same break")
        assertEquals(2, doc.pageAt(doc.text.indexOf("Page two")))
        assertEquals(1, doc.pageAt(0))
    }

    @Test
    fun `keeps urdu text intact`() {
        val xml = body("<w:p><w:r><w:t>یہ ایک اردو دستاویز ہے</w:t></w:r></w:p>")
        assertEquals("یہ ایک اردو دستاویز ہے", loaders.load(input("u.docx", docx(xml))).text.trim())
    }

    @Test
    fun `refuses an xml external entity instead of reading local files`() {
        val xxe = """<?xml version="1.0"?>
<!DOCTYPE d [<!ENTITY secret SYSTEM "file:///etc/hostname">]>
<w:document xmlns:w="$W"><w:body><w:p><w:r><w:t>&secret;</w:t></w:r></w:p></w:body></w:document>"""
        assertFailsWith<java.io.IOException> { loaders.load(input("evil.docx", docx(xxe))) }
    }

    @Test
    fun `stops a zip bomb at the expanded size limit`() {
        // ~5 MB of text that compresses to a few KB.
        val padding = "<w:p><w:r><w:t>" + "a".repeat(5_000_000) + "</w:t></w:r></w:p>"
        val bytes = docx(body(padding))
        assertTrue(bytes.size < 100_000, "fixture should be small on disk (${bytes.size})")
        assertFailsWith<DocumentTooLargeException> {
            loaders.load(input("bomb.docx", bytes), LoadLimits(maxExpandedBytes = 1_000_000))
        }
    }

    @Test
    fun `a zip without a word body is reported, not crashed on`() {
        val out = ByteArrayOutputStream()
        ZipOutputStream(out).use { it.putNextEntry(ZipEntry("other.txt")); it.write(1) }
        assertFailsWith<DocumentLoadException> { loaders.load(input("fake.docx", out.toByteArray())) }
    }

    @Test
    fun `garbage bytes surface as an IOException`() {
        assertFailsWith<java.io.IOException> { loaders.load(input("junk.docx", ByteArray(64) { it.toByte() })) }
    }
}

class SearchTextTest {

    @Test
    fun `removes urdu diacritics so vowelled and plain spellings match`() {
        assertEquals(SearchText.normalize("کتاب"), SearchText.normalize("کِتَاب"))
    }

    @Test
    fun `folds arabic letter variants onto urdu forms`() {
        // Arabic yeh + Arabic kaf, as typed on an Arabic keyboard.
        assertEquals(SearchText.normalize("کیا"), SearchText.normalize("كيا"))
        assertEquals(SearchText.normalize("یہ"), SearchText.normalize("يه"))
    }

    @Test
    fun `folds urdu digits and presentation forms`() {
        assertEquals("2024", SearchText.normalize("۲۰۲۴"))
        assertEquals(SearchText.normalize("ب"), SearchText.normalize("ﺏ"))
    }

    @Test
    fun `drops tatweel and zero width characters`() {
        assertEquals(SearchText.normalize("کتاب"), SearchText.normalize("کتـــاب"))
        assertEquals(SearchText.normalize("نہیں"), SearchText.normalize("نہ‌یں"))
    }

    @Test
    fun `lowercases latin text`() {
        assertEquals("cs4102 notes", SearchText.normalize("CS4102 Notes"))
    }
}
