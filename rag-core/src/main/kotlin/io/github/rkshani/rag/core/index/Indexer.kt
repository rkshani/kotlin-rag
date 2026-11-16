package io.github.rkshani.rag.core.index

import io.github.rkshani.rag.core.DocumentCatalog
import io.github.rkshani.rag.core.Embedder
import io.github.rkshani.rag.core.EmbeddingKind
import io.github.rkshani.rag.core.KeywordIndex
import io.github.rkshani.rag.core.RagDispatchers
import io.github.rkshani.rag.core.VectorStore
import io.github.rkshani.rag.core.chunking.Chunker
import io.github.rkshani.rag.core.loader.DocumentTooLargeException
import io.github.rkshani.rag.core.model.EmbeddedChunk
import io.github.rkshani.rag.core.model.SourceDocument
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

sealed interface IndexProgress {
    val docId: String

    /** Reading the file — page by page for PDFs and OCR. */
    data class Loading(override val docId: String, val done: Int, val total: Int) : IndexProgress {
        val fraction: Float get() = if (total <= 0) 0f else done.toFloat() / total
    }

    data class Chunking(override val docId: String, val characters: Int) : IndexProgress

    data class Embedding(override val docId: String, val done: Int, val total: Int) : IndexProgress {
        val fraction: Float get() = if (total == 0) 1f else done.toFloat() / total
    }

    data class Completed(override val docId: String, val chunks: Int, val elapsedMillis: Long) : IndexProgress
}

data class IndexConfig(
    /** Chunks embedded and written per step. Also the granularity of progress and cancellation. */
    val batchSize: Int = 16,
    /** Refuse documents larger than this instead of running out of memory halfway. */
    val maxDocumentChars: Int = 5_000_000,
    val maxChunks: Int = 50_000,
) {
    init {
        require(batchSize > 0 && maxDocumentChars > 0 && maxChunks > 0) { "index limits must be positive" }
    }
}

/**
 * Document in, searchable chunks out: chunk -> embed -> store, in batches.
 *
 * Collect the flow in a coroutine you can cancel (WorkManager, a ViewModel
 * scope). Cancelling stops at the next batch boundary and removes whatever was
 * written, so the store never keeps half a document that looks whole. If the
 * process is killed instead, the catalog entry is left marked incomplete.
 */
class Indexer(
    private val chunker: Chunker,
    private val embedder: Embedder,
    private val vectorStore: VectorStore,
    private val keywordIndex: KeywordIndex? = null,
    private val catalog: DocumentCatalog? = null,
    private val config: IndexConfig = IndexConfig(),
    private val dispatchers: RagDispatchers = RagDispatchers(),
    private val clock: () -> Long = System::currentTimeMillis,
) {

    fun index(document: SourceDocument): Flow<IndexProgress> = flow {
        if (document.text.length > config.maxDocumentChars) {
            throw DocumentTooLargeException(
                "document '${document.id}' has ${document.text.length} characters, limit is ${config.maxDocumentChars}",
            )
        }
        val started = clock()
        emit(IndexProgress.Chunking(document.id, document.text.length))

        val titled = document.title
            ?.takeIf { "title" !in document.metadata }
            ?.let { document.copy(metadata = document.metadata + ("title" to it)) }
            ?: document
        val chunks = withContext(dispatchers.default) { chunker.chunk(titled) }
        if (chunks.size > config.maxChunks) {
            throw DocumentTooLargeException(
                "document '${document.id}' produced ${chunks.size} chunks, limit is ${config.maxChunks}",
            )
        }

        // Re-indexing replaces: stale chunks from a previous version of the file
        // must not survive next to the new ones.
        removeDocument(document.id)

        try {
            var done = 0
            emit(IndexProgress.Embedding(document.id, 0, chunks.size))
            for (batch in chunks.chunked(config.batchSize)) {
                currentCoroutineContext().ensureActive()
                val vectors = embedder.embed(batch.map { it.text }, EmbeddingKind.DOCUMENT)
                check(vectors.size == batch.size) {
                    "embedder returned ${vectors.size} vectors for ${batch.size} chunks"
                }
                val embedded = batch.mapIndexed { i, chunk -> EmbeddedChunk(chunk, vectors[i]) }
                vectorStore.upsert(embedded)
                keywordIndex?.index(embedded)
                done += batch.size
                emit(IndexProgress.Embedding(document.id, done, chunks.size))
            }
            if (chunks.isNotEmpty()) catalog?.markComplete(document.id)
        } catch (failure: Throwable) {
            // Cancellation lands here too. The cleanup has to run even though the
            // coroutine is already cancelled, hence NonCancellable.
            withContext(NonCancellable) { runCatching { removeDocument(document.id) } }
            throw failure
        }

        emit(IndexProgress.Completed(document.id, chunks.size, clock() - started))
    }

    suspend fun remove(docId: String) = removeDocument(docId)

    private suspend fun removeDocument(docId: String) {
        vectorStore.deleteDocument(docId)
        keywordIndex?.deleteDocument(docId)
    }
}
