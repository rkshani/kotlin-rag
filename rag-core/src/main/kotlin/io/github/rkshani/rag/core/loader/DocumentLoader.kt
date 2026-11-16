package io.github.rkshani.rag.core.loader

import io.github.rkshani.rag.core.model.SourceDocument
import java.io.IOException
import java.io.InputStream

/**
 * A document to be read, described without committing to where it lives — a
 * file, a content Uri, an asset, a download. [open] is called once per load and
 * must return a fresh stream each time.
 */
class DocumentInput(
    val id: String,
    val name: String,
    val sizeBytes: Long? = null,
    val mimeType: String? = null,
    val metadata: Map<String, String> = emptyMap(),
    val open: () -> InputStream,
) {
    val extension: String get() = name.substringAfterLast('.', "").lowercase()
    val title: String get() = name.substringAfterLast('/').substringBeforeLast('.').ifBlank { name }
}

/**
 * Hard ceilings on what a loader will read.
 *
 * Without them a single oversized or hostile file takes the process down: the
 * whole text is held in memory as one String, then again as chunks. A 3 KB .docx
 * can also legally unzip to gigabytes, so the compressed size alone proves nothing
 * — [maxExpandedBytes] bounds what comes out of the archive.
 */
data class LoadLimits(
    val maxBytes: Long = 50L * 1024 * 1024,
    val maxChars: Int = 5_000_000,
    val maxExpandedBytes: Long = 200L * 1024 * 1024,
) {
    init {
        require(maxBytes > 0 && maxChars > 0 && maxExpandedBytes > 0) { "limits must be positive" }
    }
}

class DocumentTooLargeException(message: String) : IOException(message)

class UnsupportedDocumentException(message: String) : IOException(message)

class DocumentLoadException(message: String, cause: Throwable? = null) : IOException(message, cause)

/** Page-level progress from slow loaders (PDF, OCR). Called on the loading thread. */
fun interface LoadProgress {
    fun onProgress(done: Int, total: Int)

    companion object {
        val NONE = LoadProgress { _, _ -> }
    }
}

interface DocumentLoader {
    fun supports(input: DocumentInput): Boolean

    /** Blocking: call from an IO dispatcher. [DocumentLoaders] does this for you. */
    fun load(input: DocumentInput, limits: LoadLimits): SourceDocument

    /**
     * Slow loaders override this to report progress. OCR runs at seconds per
     * page, and a 40-page scan with no progress looks like a frozen app.
     * Long loops should also check [Thread.interrupted] so cancelling stops them.
     */
    fun load(input: DocumentInput, limits: LoadLimits, progress: LoadProgress): SourceDocument = load(input, limits)
}

/** Thrown by loaders that notice the loading thread was interrupted (cancelled). */
class LoadCancelledException(message: String) : java.io.InterruptedIOException(message)

/** Picks the first loader that accepts the input. */
class DocumentLoaders(private val loaders: List<DocumentLoader>) {

    fun load(
        input: DocumentInput,
        limits: LoadLimits = LoadLimits(),
        progress: LoadProgress = LoadProgress.NONE,
    ): SourceDocument {
        input.sizeBytes?.let { size ->
            if (size > limits.maxBytes) {
                throw DocumentTooLargeException("${input.name} is $size bytes, limit is ${limits.maxBytes}")
            }
        }
        val loader = loaders.firstOrNull { it.supports(input) }
            ?: throw UnsupportedDocumentException("no loader for '${input.name}' (${input.mimeType ?: "unknown type"})")

        val document = try {
            loader.load(input, limits, progress)
        } catch (e: IOException) {
            throw e
        } catch (e: Exception) {
            // Parsers throw a zoo of runtime exceptions on malformed files; callers
            // should only ever have to handle IOException from a load.
            throw DocumentLoadException("could not read '${input.name}': ${e.message}", e)
        }
        if (document.text.length > limits.maxChars) {
            throw DocumentTooLargeException("${input.name} has ${document.text.length} characters, limit is ${limits.maxChars}")
        }
        return document
    }

    fun supports(input: DocumentInput): Boolean = loaders.any { it.supports(input) }

    companion object {
        /** Loaders that need nothing but the JDK. PDF lives in its own module. */
        fun defaults(): List<DocumentLoader> = listOf(PlainTextLoader(), DocxLoader())
    }
}

/** Fails the read once more than [limit] bytes have come through. */
internal class BoundedInputStream(
    private val source: InputStream,
    private val limit: Long,
    private val what: String,
) : InputStream() {
    private var count = 0L

    private fun advance(n: Int) {
        if (n <= 0) return
        count += n
        if (count > limit) throw DocumentTooLargeException("$what exceeds the $limit byte limit")
    }

    override fun read(): Int = source.read().also { if (it >= 0) advance(1) }

    override fun read(b: ByteArray, off: Int, len: Int): Int = source.read(b, off, len).also { advance(it) }

    override fun close() = source.close()
}
