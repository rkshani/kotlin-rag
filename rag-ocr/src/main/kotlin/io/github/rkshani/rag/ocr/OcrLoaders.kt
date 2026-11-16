package io.github.rkshani.rag.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import androidx.exifinterface.media.ExifInterface
import io.github.rkshani.rag.core.loader.DocumentInput
import io.github.rkshani.rag.core.loader.DocumentLoadException
import io.github.rkshani.rag.core.loader.DocumentLoader
import io.github.rkshani.rag.core.loader.DocumentTooLargeException
import io.github.rkshani.rag.core.loader.LoadCancelledException
import io.github.rkshani.rag.core.loader.LoadLimits
import io.github.rkshani.rag.core.loader.LoadProgress
import io.github.rkshani.rag.core.model.SourceDocument
import io.github.rkshani.rag.loader.pdf.PageTextFallback
import io.github.rkshani.rag.loader.pdf.PdfLoader
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * OCR for scanned PDF pages, plugged into [PdfLoader] as its page fallback.
 * Pages are rendered with Android's own PdfRenderer — no extra dependency.
 *
 * @param language used when the document's metadata does not say
 *   (see [OcrLanguage.METADATA_KEY]); [PdfLoader] passes no metadata to the
 *   fallback, so use [OcrPdf.loader] which wires the per-document choice.
 */
class OcrPageFallback(
    private val engine: OcrEngine,
    private val language: () -> OcrLanguage,
    private val dpi: Int = 300,
    private val maxSide: Int = 2480,
) : PageTextFallback {

    override fun open(pdf: File): PageTextFallback.Session {
        val descriptor = ParcelFileDescriptor.open(pdf, ParcelFileDescriptor.MODE_READ_ONLY)
        val renderer = try {
            PdfRenderer(descriptor)
        } catch (failure: Exception) {
            descriptor.close()
            throw DocumentLoadException("could not render PDF pages for OCR: ${failure.message}", failure)
        }
        val chosen = language()
        return object : PageTextFallback.Session {
            override fun text(pageIndex: Int): String {
                if (pageIndex !in 0 until renderer.pageCount) return ""
                val bitmap = renderer.openPage(pageIndex).use { page ->
                    val (width, height) = RenderSize.of(page.width, page.height, dpi, maxSide)
                    Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmap ->
                        // PdfRenderer draws onto a transparent bitmap. OCR engines read
                        // transparent as black, so black text vanishes into a black page.
                        bitmap.eraseColor(Color.WHITE)
                        page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                    }
                }
                return try {
                    OcrText.clean(engine.recognize(bitmap, chosen))
                } finally {
                    bitmap.recycle()
                }
            }

            override fun close() {
                renderer.close()
                descriptor.close()
            }
        }
    }
}

/** Builds a [PdfLoader] that OCRs scanned pages in each document's own language. */
object OcrPdf {
    fun loader(context: Context, engine: OcrEngine, default: OcrLanguage = OcrLanguage.ENGLISH): DocumentLoader =
        object : DocumentLoader {
            private var current = default
            private val pdf = PdfLoader(context, fallback = OcrPageFallback(engine, { current }))

            override fun supports(input: DocumentInput) = pdf.supports(input)

            override fun load(input: DocumentInput, limits: LoadLimits) = load(input, limits, LoadProgress.NONE)

            @Synchronized
            override fun load(input: DocumentInput, limits: LoadLimits, progress: LoadProgress): SourceDocument {
                current = OcrLanguage.fromMetadata(input.metadata, default)
                return pdf.load(input, limits, progress)
            }
        }
}

/**
 * Photos and screenshots of notes, whiteboards and printed pages: JPEG, PNG,
 * WebP (and HEIF on Android 9+).
 *
 * The photo is downsampled while decoding (a 50 MP photo would not fit in
 * memory at full size) and turned upright using its EXIF orientation — phones
 * store most photos sideways and set a flag, and OCR on a sideways page returns
 * nothing useful.
 */
class ImageOcrLoader(
    private val engine: OcrEngine,
    private val default: OcrLanguage = OcrLanguage.ENGLISH,
    private val maxSide: Int = 2480,
) : DocumentLoader {

    override fun supports(input: DocumentInput): Boolean =
        input.extension in EXTENSIONS || input.mimeType?.startsWith("image/") == true

    override fun load(input: DocumentInput, limits: LoadLimits): SourceDocument = load(input, limits, LoadProgress.NONE)

    override fun load(input: DocumentInput, limits: LoadLimits, progress: LoadProgress): SourceDocument {
        val bytes = readBounded(input, limits.maxBytes)
        val language = OcrLanguage.fromMetadata(input.metadata, default)
        if (Thread.interrupted()) throw LoadCancelledException("loading '${input.name}' was cancelled")

        val bitmap = decodeUpright(bytes) ?: throw DocumentLoadException("'${input.name}' is not a readable image")
        val text = try {
            OcrText.clean(engine.recognize(bitmap, language))
        } finally {
            bitmap.recycle()
        }
        progress.onProgress(1, 1)
        if (OcrText.contentChars(text) == 0) throw DocumentLoadException("no text found in '${input.name}'")
        if (text.length > limits.maxChars) throw DocumentTooLargeException("'${input.name}' produced too much text")

        return SourceDocument(
            id = input.id,
            text = text,
            title = input.title,
            metadata = input.metadata + mapOf("title" to input.title, "ocrPages" to "1"),
        )
    }

    private fun decodeUpright(bytes: ByteArray): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        val options = BitmapFactory.Options().apply {
            inSampleSize = RenderSize.sampleSize(bounds.outWidth, bounds.outHeight, maxSide)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options) ?: return null
        val rotation = runCatching { ExifInterface(ByteArrayInputStream(bytes)).rotationDegrees }.getOrDefault(0)
        if (rotation == 0) return decoded
        val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
        if (rotated != decoded) decoded.recycle()
        return rotated
    }

    private fun readBounded(input: DocumentInput, maxBytes: Long): ByteArray {
        input.sizeBytes?.let { if (it > maxBytes) throw DocumentTooLargeException("'${input.name}' is $it bytes, limit is $maxBytes") }
        val out = ByteArrayOutputStream()
        input.open().use { stream ->
            val buffer = ByteArray(64 * 1024)
            var total = 0L
            while (true) {
                val read = stream.read(buffer)
                if (read < 0) break
                total += read
                if (total > maxBytes) throw DocumentTooLargeException("'${input.name}' exceeds the $maxBytes byte limit")
                out.write(buffer, 0, read)
            }
        }
        return out.toByteArray()
    }

    private companion object {
        val EXTENSIONS = setOf("jpg", "jpeg", "png", "webp", "heic", "heif", "bmp")
    }
}
