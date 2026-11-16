package io.github.rkshani.rag.core.retrieval

import io.github.rkshani.rag.core.Embedder
import io.github.rkshani.rag.core.EmbeddingKind
import io.github.rkshani.rag.core.KeywordIndex
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.VectorStore
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope

/**
 * Retrieval over both a [VectorStore] and a [KeywordIndex], fused with RRF.
 *
 * Neither half is sufficient alone. Embeddings capture meaning but blur exact
 * literals — a course code, a function name, a chemical formula, a rare proper
 * noun all embed into roughly the same neighbourhood as their surrounding prose,
 * so a query for "CS4102" retrieves generically related text instead of the one
 * chunk that names it. BM25 nails those and is useless for paraphrase. Running
 * both and fusing is the cheapest large quality win available to a RAG system.
 *
 * @param keywordIndex optional — pass null for vector-only retrieval.
 */
class HybridRetriever(
    private val embedder: Embedder,
    private val vectorStore: VectorStore,
    private val keywordIndex: KeywordIndex? = null,
    private val config: HybridRetrievalConfig = HybridRetrievalConfig(),
    /** Notified when one source fails and retrieval degrades to the other. */
    private val onSourceFailure: (RetrievalSource, Throwable) -> Unit = { _, _ -> },
    /** Optional second stage over the fused top [HybridRetrievalConfig.rerankCandidates]. */
    private val reranker: Reranker? = null,
    private val onRerankFailure: (Throwable) -> Unit = {},
) : Retriever {

    override suspend fun retrieve(query: String, k: Int, filter: SearchFilter?): List<RetrievedChunk> {
        if (query.isBlank() || k <= 0) return emptyList()

        val lists = coroutineScope {
            val vectorDeferred = async {
                catching {
                    val queryVector = embedder.embed(query, EmbeddingKind.QUERY)
                    vectorStore.search(queryVector, config.vectorCandidates, filter)
                }
            }
            val keywordDeferred = async {
                keywordIndex?.let {
                    catching { it.search(query, config.keywordCandidates, filter) }
                }
            }

            val vectorResult = vectorDeferred.await()
            val keywordResult = keywordDeferred.await()

            // Degrade rather than fail: one working source still answers the
            // question. Only when everything available fails do we surface it.
            vectorResult.exceptionOrNull()?.let { onSourceFailure(RetrievalSource.VECTOR, it) }
            keywordResult?.exceptionOrNull()?.let { onSourceFailure(RetrievalSource.KEYWORD, it) }

            if (vectorResult.isFailure && keywordResult?.isSuccess != true) {
                throw vectorResult.exceptionOrNull()
                    ?: keywordResult?.exceptionOrNull()
                    ?: IllegalStateException("retrieval failed with no cause")
            }

            buildList {
                vectorResult.getOrNull()?.let {
                    add(RankedList(RetrievalSource.VECTOR, it, config.vectorWeight))
                }
                keywordResult?.getOrNull()?.let {
                    add(RankedList(RetrievalSource.KEYWORD, it, config.keywordWeight))
                }
            }
        }

        val fused = ReciprocalRankFusion.fuse(lists, config.rrfK)
        return applyDiversityCap(rerank(query, fused), k)
    }

    /**
     * Re-scores the head of the fused list and leaves the tail in fused order
     * behind it. A failing re-ranker never fails the query — fused order is
     * already a perfectly good answer.
     */
    private suspend fun rerank(query: String, fused: List<RetrievedChunk>): List<RetrievedChunk> {
        val model = reranker ?: return fused
        if (fused.size < 2) return fused
        val head = fused.take(config.rerankCandidates)
        val reordered = try {
            model.rerank(query, head)
        } catch (cancelled: kotlinx.coroutines.CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            onRerankFailure(failure)
            return fused
        }
        return reordered + fused.drop(head.size)
    }

    /**
     * Greedy pass in fused-score order, skipping documents that have already
     * contributed their quota. Greedy is correct here because the list is
     * already sorted: dropping a lower-scored chunk can never promote a
     * higher-scored one above where it sits.
     */
    private fun applyDiversityCap(fused: List<RetrievedChunk>, k: Int): List<RetrievedChunk> {
        if (config.maxPerDocument == Int.MAX_VALUE) return fused.take(k)

        val perDocument = HashMap<String, Int>()
        val out = ArrayList<RetrievedChunk>(k)
        for (candidate in fused) {
            val used = perDocument.getOrDefault(candidate.chunk.docId, 0)
            if (used >= config.maxPerDocument) continue
            perDocument[candidate.chunk.docId] = used + 1
            out.add(candidate)
            if (out.size == k) break
        }
        return out
    }
}

/**
 * `runCatching` that lets cancellation through.
 *
 * Plain `runCatching` also catches CancellationException, which turns "the user
 * left the screen" into "a source failed" and keeps the query running.
 */
private inline fun <T> catching(block: () -> T): Result<T> = try {
    Result.success(block())
} catch (cancelled: kotlinx.coroutines.CancellationException) {
    throw cancelled
} catch (failure: Throwable) {
    Result.failure(failure)
}

/** Vector-only retrieval, for corpora where lexical matching adds nothing. */
class VectorRetriever(
    private val embedder: Embedder,
    private val vectorStore: VectorStore,
) : Retriever {
    override suspend fun retrieve(query: String, k: Int, filter: SearchFilter?): List<RetrievedChunk> {
        if (query.isBlank() || k <= 0) return emptyList()
        val queryVector = embedder.embed(query, EmbeddingKind.QUERY)
        return vectorStore.search(queryVector, k, filter).mapIndexed { index, scored ->
            RetrievedChunk(
                chunk = scored.chunk,
                score = scored.score,
                ranks = mapOf(RetrievalSource.VECTOR to index + 1),
                sourceScores = mapOf(RetrievalSource.VECTOR to scored.score),
            )
        }
    }
}
