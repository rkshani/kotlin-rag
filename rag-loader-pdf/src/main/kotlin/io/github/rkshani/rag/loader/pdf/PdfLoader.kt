package io.github.rkshani.rag.loader.pdf

import android.content.Context
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.io.MemoryUsageSetting
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.encryption.InvalidPasswordException
import com.tom_roush.pdfbox.text.PDFTextStripper
import io.github.rkshani.rag.core.loader.DocumentInput
import io.github.rkshani.rag.core.loader.DocumentLoadException
import io.github.rkshani.rag.core.loader.DocumentLoader
import io.github.rkshani.rag.core.loader.DocumentTooLargeException
import io.github.rkshani.rag.core.loader.LoadCancelledException
import io.github.rkshani.rag.core.loader.LoadLimits
import io.github.rkshani.rag.core.loader.LoadProgress
import io.github.rkshani.rag.core.model.SourceDocument
import java.io.File

/**
 * Text for pages that have no usable text layer — scanned pages. The OCR module
 * implements this; the PDF loader only knows "this page needs help".
 *
 * [open] is called once per document with the PDF already on disk, so an
 * implementation can keep one renderer open across all its pages.
 */
interface PageTextFallback {
    fun open(pdf: File): Session

    interface Session : AutoCloseable {
        /** Text for the 0-based [pageIndex]. Blocking; may take seconds. */
        fun text(pageIndex: Int): String
    }
}

/**
 * PDF loader on PdfBox-Android, with a page offset recorded for every page so
 * citations can say "page 7".
 *
 * - The PDF is copied to a temp file first and parsed with temp-file buffering:
 *   PdfBox needs random access, and holding a 50 MB PDF in the heap plus
 *   PdfBox's own structures is how phones run out of memory.
 * - Copying is byte-bounded by [LoadLimits.maxBytes]; [maxPages] bounds parsing.
 * - Pages with less than [minPageChars] characters of text are handed to
 *   [fallback] (OCR) when one is set. This is per page, so a mostly digital PDF
 *   with a few scanned pages works, and digital pages never pay for OCR.
 * - Without a fallback, an image-only PDF fails with a clear message.
 * - Reports page progress and stops at the next page when the thread is
 *   interrupted (the indexing coroutine was cancelled).
 */
class PdfLoader(
    context: Context,
    private val fallback: PageTextFallback? = null,
    private val maxPages: Int = 2_000,
    private val minPageChars: Int = 20,
) : DocumentLoader {

    private val appContext = context.applicationContext

    init {
        PDFBoxResourceLoader.init(appContext)
    }

    override fun supports(input: DocumentInput): Boolean =
        input.extension == "pdf" || input.mimeType == "application/pdf"

    override fun load(input: DocumentInput, limits: LoadLimits): SourceDocument = load(input, limits, LoadProgress.NONE)

    override fun load(input: DocumentInput, limits: LoadLimits, progress: LoadProgress): SourceDocument {
        val temp = File.createTempFile("rag-pdf-", ".pdf", appContext.cacheDir)
        try {
            copyBounded(input, temp, limits.maxBytes)
            val document = try {
                PDDocument.load(temp, MemoryUsageSetting.setupTempFileOnly())
            } catch (locked: InvalidPasswordException) {
                throw DocumentLoadException("'${input.name}' is password protected")
            }
            return document.use { pdf -> extract(input, pdf, temp, limits, progress) }
        } finally {
            temp.delete()
        }
    }

    private fun extract(input: DocumentInput, pdf: PDDocument, file: File, limits: LoadLimits, progress: LoadProgress): SourceDocument {
        val pageCount = pdf.numberOfPages
        if (pageCount > maxPages) throw DocumentTooLargeException("'${input.name}' has $pageCount pages, limit is $maxPages")

        val stripper = PDFTextStripper().apply { sortByPosition = true }
        val text = StringBuilder()
        val pageOffsets = ArrayList<Int>(pageCount)
        var ocrPages = 0
        var session: PageTextFallback.Session? = null
        try {
            for (page in 1..pageCount) {
                if (Thread.interrupted()) throw LoadCancelledException("loading '${input.name}' was cancelled")
                pageOffsets.add(text.length)
                stripper.startPage = page
                stripper.endPage = page
                var pageText = stripper.getText(pdf)

                if (fallback != null && pageText.count { it.isLetterOrDigit() } < minPageChars) {
                    val ocr = (session ?: fallback.open(file).also { session = it }).text(page - 1)
                    if (ocr.isNotBlank()) {
                        pageText = ocr
                        ocrPages++
                    }
                }
                if (text.length + pageText.length > limits.maxChars) {
                    throw DocumentTooLargeException("'${input.name}' has more than ${limits.maxChars} characters of text")
                }
                text.append(pageText.trimEnd()).append("\n\n")
                progress.onProgress(page, pageCount)
            }
        } finally {
            session?.close()
        }

        if (text.isBlank()) {
            val hint = if (fallback == null) " — it looks scanned; add the OCR module" else " even after OCR"
            throw DocumentLoadException("'${input.name}' has no readable text$hint")
        }
        val title = pdf.documentInformation?.title?.takeIf { it.isNotBlank() } ?: input.title
        return SourceDocument(
            id = input.id,
            text = text.toString(),
            title = title,
            pageOffsets = pageOffsets,
            metadata = input.metadata + mapOf(
                "title" to title,
                "pages" to pageCount.toString(),
                "ocrPages" to ocrPages.toString(),
            ),
        )
    }

    private fun copyBounded(input: DocumentInput, target: File, maxBytes: Long) {
        input.open().use { source ->
            target.outputStream().use { sink ->
                val buffer = ByteArray(64 * 1024)
                var total = 0L
                while (true) {
                    val read = source.read(buffer)
                    if (read < 0) break
                    total += read
                    if (total > maxBytes) throw DocumentTooLargeException("'${input.name}' exceeds the $maxBytes byte limit")
                    sink.write(buffer, 0, read)
                }
            }
        }
    }
}
