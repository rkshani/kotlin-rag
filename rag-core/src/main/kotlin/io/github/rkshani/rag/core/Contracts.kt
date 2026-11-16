package io.github.rkshani.rag.core

import io.github.rkshani.rag.core.model.EmbeddedChunk
import io.github.rkshani.rag.core.model.ScoredChunk
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

/**
 * The dispatchers the library runs on, in one injectable place.
 *
 * Hardcoded `Dispatchers.IO` / `Dispatchers.Default` inside classes cannot be
 * replaced in tests, which makes timing-dependent code untestable. Every
 * component takes one of these instead; production code uses the defaults.
 */
data class RagDispatchers(
    val io: CoroutineDispatcher = Dispatchers.IO,
    val default: CoroutineDispatcher = Dispatchers.Default,
)

/**
 * Some embedding models (E5, BGE, GTE) are asymmetric and expect a different
 * prefix for the stored passage vs the incoming question. MiniLM is symmetric
 * and ignores this, but the contract has to allow for it or swapping models later
 * silently degrades retrieval quality.
 */
enum class EmbeddingKind { DOCUMENT, QUERY }

interface Embedder : AutoCloseable {
    /** Output vector size. Must match what the [VectorStore] was created with. */
    val dimensions: Int

    /** Tokens the model accepts before truncation. Chunker must stay under this. */
    val maxSequenceTokens: Int

    /**
     * Embeds a batch. Implementations should honour batching internally rather
     * than looping one text at a time — session overhead dominates on-device.
     * Returned vectors are L2-normalised, so cosine == dot product.
     */
    suspend fun embed(texts: List<String>, kind: EmbeddingKind = EmbeddingKind.DOCUMENT): List<FloatArray>

    suspend fun embed(text: String, kind: EmbeddingKind = EmbeddingKind.DOCUMENT): FloatArray =
        embed(listOf(text), kind).first()
}

/** Filter applied before scoring. Null fields mean "no constraint". */
data class SearchFilter(
    val docIds: Set<String>? = null,
    val metadataEquals: Map<String, String> = emptyMap(),
)

interface VectorStore {
    /**
     * Vector size this store holds, when it is fixed. Lets a builder reject an
     * embedder of the wrong size at startup instead of at the first search.
     */
    val dimensions: Int? get() = null

    suspend fun upsert(chunks: List<EmbeddedChunk>)

    suspend fun search(query: FloatArray, k: Int, filter: SearchFilter? = null): List<ScoredChunk>

    suspend fun deleteDocument(docId: String)

    suspend fun count(): Int
}

/** Lexical half of hybrid retrieval (Phase 4). Backed by SQLite FTS on Android. */
interface KeywordIndex {
    suspend fun index(chunks: List<EmbeddedChunk>)
    suspend fun search(query: String, k: Int, filter: SearchFilter? = null): List<ScoredChunk>
    suspend fun deleteDocument(docId: String)
}

data class GenerationOptions(
    val maxTokens: Int = 512,
    val temperature: Float = 0.2f,
    val topK: Int = 40,
)

interface Generator : AutoCloseable {
    /**
     * True when the model file is present AND the device can actually run it.
     * Callers must check this and fall back to extractive answers when false —
     * a 2B model will OOM on a 3 GB device.
     */
    suspend fun isAvailable(): Boolean

    /**
     * True when the engine wraps the prompt in the model's chat template itself
     * (LiteRT-LM conversations do). The prompt must then be sent plain —
     * templating it twice puts literal control tokens in front of the model.
     */
    val appliesChatTemplate: Boolean get() = false

    /** Emits partial text as it is produced. */
    fun generate(prompt: String, options: GenerationOptions = GenerationOptions()): Flow<String>
}

/** What the store knows about one indexed document. */
data class DocumentInfo(
    val docId: String,
    val title: String?,
    val chunkCount: Int,
    val indexedAtMillis: Long,
    /**
     * False while indexing is in flight, and stays false if the process died
     * mid-index. Such a document is searchable but partial — re-index it.
     */
    val complete: Boolean,
    val metadata: Map<String, String> = emptyMap(),
)

/** Listing and bookkeeping for indexed documents — what a "my documents" screen needs. */
interface DocumentCatalog {
    suspend fun documents(): List<DocumentInfo>
    suspend fun document(docId: String): DocumentInfo?

    /** Called by the indexer once every chunk of [docId] has been stored. */
    suspend fun markComplete(docId: String)
}
