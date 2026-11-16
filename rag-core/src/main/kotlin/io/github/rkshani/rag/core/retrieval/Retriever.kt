package io.github.rkshani.rag.core.retrieval

import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.ScoredChunk

enum class RetrievalSource { VECTOR, KEYWORD }

/**
 * A fused result, carrying where it came from.
 *
 * Provenance is not decoration: when retrieval goes wrong, the only useful
 * question is whether the vector side or the keyword side surfaced a bad chunk,
 * and a bare score cannot answer it. The demo UI shows these ranks directly.
 */
data class RetrievedChunk(
    val chunk: Chunk,
    val score: Float,
    /** 1-based rank in each list that returned this chunk. Absent means it missed. */
    val ranks: Map<RetrievalSource, Int>,
    /** The raw, uncalibrated score each source assigned. */
    val sourceScores: Map<RetrievalSource, Float>,
    /** Set when a [Reranker] scored this chunk; [score] then holds the same value. */
    val rerankScore: Float? = null,
) {
    val foundByBoth: Boolean get() = ranks.size > 1
}

/** One source's ordered output, ready for fusion. */
data class RankedList(
    val source: RetrievalSource,
    val results: List<ScoredChunk>,
    val weight: Double = 1.0,
)

data class HybridRetrievalConfig(
    /**
     * How deep to search each source before fusing. Larger than the requested k
     * on purpose: fusion can only promote what it was given, so a chunk ranked
     * 15th by vectors and 2nd by keywords is only reachable if both lists run deep.
     */
    val vectorCandidates: Int = 20,
    val keywordCandidates: Int = 20,
    /**
     * RRF smoothing constant. At the standard 60, the gap between rank 1 and
     * rank 2 is small, so agreement across sources outweighs a single source's
     * confidence — which is the whole point of using ranks over raw scores.
     */
    val rrfK: Int = ReciprocalRankFusion.DEFAULT_K,
    val vectorWeight: Double = 1.0,
    val keywordWeight: Double = 1.0,
    /**
     * Caps results from any one document. Without this, a query that matches one
     * long document well returns five near-identical chunks from it and the
     * generator sees no breadth. [Int.MAX_VALUE] disables the cap.
     */
    val maxPerDocument: Int = Int.MAX_VALUE,
    /**
     * How many fused results a [Reranker] is asked to re-score. A cross-encoder
     * reads query and passage together, so its cost is one model pass per
     * candidate — on a phone this is the knob that decides whether re-ranking
     * takes 200 ms or 2 s.
     */
    val rerankCandidates: Int = 10,
) {
    init {
        require(vectorCandidates > 0 && keywordCandidates > 0) { "candidate counts must be positive" }
        require(maxPerDocument > 0) { "maxPerDocument must be positive" }
        require(rerankCandidates > 0) { "rerankCandidates must be positive" }
    }
}

interface Retriever {
    suspend fun retrieve(
        query: String,
        k: Int = 5,
        filter: SearchFilter? = null,
    ): List<RetrievedChunk>
}

/**
 * Second-stage scoring of a short candidate list.
 *
 * First-stage retrieval embeds the query and the passage separately, so it can
 * only compare two summaries. A re-ranker reads both texts together and can
 * tell "mentions the topic" from "answers the question". It is too slow to run
 * over a whole index, which is why it only sees the fused top few.
 *
 * Returns the candidates in their new order, best first.
 */
fun interface Reranker {
    suspend fun rerank(query: String, candidates: List<RetrievedChunk>): List<RetrievedChunk>
}
