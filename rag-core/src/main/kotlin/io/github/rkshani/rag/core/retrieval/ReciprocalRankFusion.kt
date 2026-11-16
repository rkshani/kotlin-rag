package io.github.rkshani.rag.core.retrieval

/**
 * Reciprocal Rank Fusion.
 *
 *     score(chunk) = Σ over sources  weight / (k + rank)
 *
 * with rank 1-based within each source.
 *
 * The reason to fuse ranks rather than scores: cosine similarity lives in 0..1
 * and clusters tightly (0.72 vs 0.68 is a meaningful gap), while BM25 is
 * unbounded and depends on corpus statistics. Any attempt to put them on a
 * common scale needs calibration that drifts as the corpus grows. Ranks need no
 * calibration at all, and a chunk both sources rank highly beats one that a
 * single source loves — which is usually the right answer.
 */
object ReciprocalRankFusion {

    /**
     * Standard value from the original RRF paper. Large relative to typical rank
     * positions, which deliberately flattens the curve: it makes the difference
     * between rank 1 and rank 3 modest, so agreement matters more than a single
     * source's ordering. Lower it to trust individual rankings more.
     */
    const val DEFAULT_K: Int = 60

    fun fuse(
        lists: List<RankedList>,
        k: Int = DEFAULT_K,
        limit: Int = Int.MAX_VALUE,
    ): List<RetrievedChunk> {
        require(k > 0) { "rrfK must be positive" }
        if (lists.isEmpty() || limit <= 0) return emptyList()

        // LinkedHashMap so that chunks tied on fused score keep the order of the
        // first source that produced them — otherwise results reshuffle between
        // identical queries, which looks like a bug to users.
        val accumulator = LinkedHashMap<String, Accumulator>()

        lists.forEach { list ->
            list.results.forEachIndexed { index, scored ->
                val rank = index + 1
                val entry = accumulator.getOrPut(scored.chunk.id) { Accumulator(scored) }
                entry.score += list.weight / (k + rank)
                entry.ranks[list.source] = rank
                entry.sourceScores[list.source] = scored.score
            }
        }

        return accumulator.values
            .map { it.toRetrievedChunk() }
            .sortedByDescending { it.score }
            .take(limit)
    }

    private class Accumulator(private val scored: io.github.rkshani.rag.core.model.ScoredChunk) {
        var score: Double = 0.0
        val ranks = LinkedHashMap<RetrievalSource, Int>()
        val sourceScores = LinkedHashMap<RetrievalSource, Float>()

        fun toRetrievedChunk() = RetrievedChunk(
            chunk = scored.chunk,
            score = score.toFloat(),
            ranks = ranks.toMap(),
            sourceScores = sourceScores.toMap(),
        )
    }
}
