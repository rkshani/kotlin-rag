package io.github.rkshani.rag.core

import io.github.rkshani.rag.core.answer.Answer
import io.github.rkshani.rag.core.answer.AnswerEvent
import io.github.rkshani.rag.core.answer.ContextAssembler
import io.github.rkshani.rag.core.answer.ContextOrder
import io.github.rkshani.rag.core.answer.InjectionPolicy
import io.github.rkshani.rag.core.answer.PromptBuilder
import io.github.rkshani.rag.core.answer.PromptTemplate
import io.github.rkshani.rag.core.answer.RagAnswerer
import io.github.rkshani.rag.core.chat.ChatSession
import io.github.rkshani.rag.core.chat.FollowUpQueryRewriter
import io.github.rkshani.rag.core.chat.QueryRewriter
import io.github.rkshani.rag.core.chunking.Chunker
import io.github.rkshani.rag.core.chunking.RecursiveChunker
import io.github.rkshani.rag.core.chunking.TokenEstimator
import io.github.rkshani.rag.core.index.IndexConfig
import io.github.rkshani.rag.core.index.IndexProgress
import io.github.rkshani.rag.core.index.Indexer
import io.github.rkshani.rag.core.loader.DocumentInput
import io.github.rkshani.rag.core.loader.DocumentLoader
import io.github.rkshani.rag.core.loader.DocumentLoaders
import io.github.rkshani.rag.core.loader.LoadLimits
import io.github.rkshani.rag.core.model.SourceDocument
import io.github.rkshani.rag.core.retrieval.HybridRetrievalConfig
import io.github.rkshani.rag.core.retrieval.HybridRetriever
import io.github.rkshani.rag.core.retrieval.Reranker
import io.github.rkshani.rag.core.retrieval.RetrievedChunk
import io.github.rkshani.rag.core.retrieval.Retriever
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runInterruptible
import kotlin.math.max
import kotlin.math.min

/**
 * The library's front door: index documents, search them, ask questions.
 *
 * ```
 * val rag = KotlinRag.builder()
 *     .embedder(embedder)
 *     .store(roomStore)          // or InMemoryStore()
 *     .generator(generator)      // optional — without it answers are extractive
 *     .build()
 *
 * rag.index(input).collect { progress -> ... }
 * rag.ask("What does chapter 3 say?").collect { event -> ... }
 * ```
 *
 * Owns the embedder, generator and reranker it was given and closes them in [close].
 */
class KotlinRag private constructor(
    val retriever: Retriever,
    private val indexer: Indexer,
    private val answerer: RagAnswerer,
    private val catalog: DocumentCatalog?,
    private val loaders: DocumentLoaders,
    private val loadLimits: LoadLimits,
    private val dispatchers: RagDispatchers,
    private val owned: List<AutoCloseable>,
) : AutoCloseable {

    /** Indexes text you already have. Re-indexing the same id replaces the old version. */
    fun index(document: SourceDocument): Flow<IndexProgress> = indexer.index(document)

    /**
     * Reads the file with the first loader that supports it, then indexes it.
     *
     * Loading runs interruptibly: cancelling the collector interrupts the
     * loading thread, so a long OCR job stops at the next page instead of
     * running to the end in the background.
     */
    fun index(input: DocumentInput): Flow<IndexProgress> = channelFlow {
        val document = try {
            runInterruptible(dispatchers.io) {
                loaders.load(input, loadLimits) { done, total -> trySend(IndexProgress.Loading(input.id, done, total)) }
            }
        } catch (interrupted: java.io.InterruptedIOException) {
            // A loader that noticed the interrupt reports it as an IOException.
            // If we were cancelled, surface that as cancellation, not a failure.
            currentCoroutineContext().ensureActive()
            throw interrupted
        }
        indexer.index(document).collect { send(it) }
    }

    fun canLoad(input: DocumentInput): Boolean = loaders.supports(input)

    suspend fun search(query: String, k: Int = 5, filter: SearchFilter? = null): List<RetrievedChunk> =
        retriever.retrieve(query, k, filter)

    /** A single question with no conversation history. */
    fun ask(question: String, filter: SearchFilter? = null): Flow<AnswerEvent> = answerer.answer(question, filter)

    /** [ask], waiting for the finished answer instead of streaming. */
    suspend fun askAndWait(question: String, filter: SearchFilter? = null): Answer =
        ask(question, filter).filterIsInstance<AnswerEvent.Completed>().first().answer

    /** A conversation that remembers earlier turns and resolves follow-up questions. */
    fun chat(maxTurns: Int = 6): ChatSession = ChatSession(answerer, maxTurns)

    suspend fun documents(): List<DocumentInfo> = catalog?.documents().orEmpty()

    suspend fun document(docId: String): DocumentInfo? = catalog?.document(docId)

    suspend fun remove(docId: String) = indexer.remove(docId)

    override fun close() {
        owned.forEach { runCatching { it.close() } }
    }

    companion object {
        fun builder(): Builder = Builder()
    }

    class Builder internal constructor() {
        private var embedder: Embedder? = null
        private var vectorStore: VectorStore? = null
        private var keywordIndex: KeywordIndex? = null
        private var catalog: DocumentCatalog? = null
        private var generator: Generator? = null
        private var reranker: Reranker? = null
        private var chunker: Chunker? = null
        private var tokenEstimator: TokenEstimator? = null
        private var loaders: MutableList<DocumentLoader> = DocumentLoaders.defaults().toMutableList()
        private var loadLimits = LoadLimits()
        private var indexConfig = IndexConfig()
        private var retrievalConfig = HybridRetrievalConfig()
        private var answerConfig = RagAnswerer.AnswerConfig()
        private var promptTemplate: PromptTemplate = PromptTemplate.Gemma
        private var systemInstruction: String = PromptBuilder.DEFAULT_SYSTEM_INSTRUCTION
        private var contextTokens = 1200
        private var injectionPolicy = InjectionPolicy.WITHHOLD
        private var queryRewriter: QueryRewriter = FollowUpQueryRewriter()
        private var dispatchers = RagDispatchers()
        private var onFailure: (String, Throwable) -> Unit = { _, _ -> }

        fun embedder(value: Embedder) = apply { embedder = value }

        fun vectorStore(value: VectorStore) = apply { vectorStore = value }

        fun keywordIndex(value: KeywordIndex?) = apply { keywordIndex = value }

        fun catalog(value: DocumentCatalog?) = apply { catalog = value }

        /** One object that is all three — Room's store set, or [io.github.rkshani.rag.core.store.InMemoryStore]. */
        fun <T> store(value: T) where T : VectorStore, T : KeywordIndex, T : DocumentCatalog = apply {
            vectorStore = value
            keywordIndex = value
            catalog = value
        }

        fun generator(value: Generator?) = apply { generator = value }

        fun reranker(value: Reranker?) = apply { reranker = value }

        fun chunker(value: Chunker) = apply { chunker = value }

        /** Used by the default chunker. Pass the embedder's real tokenizer count for exact chunk sizes. */
        fun tokenEstimator(value: TokenEstimator) = apply { tokenEstimator = value }

        fun addLoader(value: DocumentLoader) = apply { loaders.add(0, value) }

        fun loadLimits(value: LoadLimits) = apply { loadLimits = value }

        fun indexConfig(value: IndexConfig) = apply { indexConfig = value }

        fun retrievalConfig(value: HybridRetrievalConfig) = apply { retrievalConfig = value }

        fun answerConfig(value: RagAnswerer.AnswerConfig) = apply { answerConfig = value }

        fun promptTemplate(value: PromptTemplate) = apply { promptTemplate = value }

        fun systemInstruction(value: String) = apply { systemInstruction = value }

        fun contextTokens(value: Int) = apply { contextTokens = value }

        fun injectionPolicy(value: InjectionPolicy) = apply { injectionPolicy = value }

        fun queryRewriter(value: QueryRewriter) = apply { queryRewriter = value }

        fun dispatchers(value: RagDispatchers) = apply { dispatchers = value }

        /** Degraded-but-working events: a retrieval source, the reranker or generation failed. */
        fun onFailure(listener: (component: String, error: Throwable) -> Unit) = apply { onFailure = listener }

        fun build(): KotlinRag {
            val embedder = requireNotNull(embedder) { "an embedder is required" }
            val vectorStore = requireNotNull(vectorStore) { "a vector store is required" }
            vectorStore.dimensions?.let { stored ->
                require(stored == embedder.dimensions) {
                    "store holds $stored-dim vectors but the embedder produces ${embedder.dimensions} — " +
                        "a different model needs a fresh store"
                }
            }

            val chunker = chunker ?: defaultChunker(embedder)
            val retriever = HybridRetriever(
                embedder = embedder,
                vectorStore = vectorStore,
                keywordIndex = keywordIndex,
                config = retrievalConfig,
                onSourceFailure = { source, error -> onFailure("retrieval:${source.name.lowercase()}", error) },
                reranker = reranker,
                onRerankFailure = { onFailure("reranker", it) },
            )
            val answerer = RagAnswerer(
                retriever = retriever,
                generator = generator,
                assembler = ContextAssembler(
                    tokenBudget = contextTokens,
                    order = ContextOrder.EDGES,
                    injectionPolicy = injectionPolicy,
                ),
                promptBuilder = PromptBuilder(promptTemplate, systemInstruction),
                config = answerConfig,
                queryRewriter = queryRewriter,
                onGenerationFailure = { onFailure("generator", it) },
            )
            val indexer = Indexer(
                chunker = chunker,
                embedder = embedder,
                vectorStore = vectorStore,
                keywordIndex = keywordIndex,
                catalog = catalog,
                config = indexConfig.copy(maxDocumentChars = min(indexConfig.maxDocumentChars, loadLimits.maxChars)),
                dispatchers = dispatchers,
            )
            val owned = listOfNotNull(embedder, generator, reranker as? AutoCloseable)
            return KotlinRag(retriever, indexer, answerer, catalog, DocumentLoaders(loaders), loadLimits, dispatchers, owned)
        }

        /**
         * Chunks must fit the embedder's window: text past it is silently cut
         * off and never embedded. MiniLM's window is 256 tokens, so the old
         * 400-token default lost a third of every chunk.
         */
        private fun defaultChunker(embedder: Embedder): Chunker {
            val target = min(400, (embedder.maxSequenceTokens * 0.85).toInt()).coerceAtLeast(32)
            val overlap = max(8, target / 8)
            return tokenEstimator?.let { RecursiveChunker(target, overlap, tokenEstimator = it) }
                ?: RecursiveChunker(target, overlap)
        }
    }
}
