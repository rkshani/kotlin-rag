package io.github.rkshani.rag.core.answer

import io.github.rkshani.rag.core.chunking.HeuristicTokenEstimator
import io.github.rkshani.rag.core.chunking.TokenEstimator
import io.github.rkshani.rag.core.retrieval.RetrievedChunk

enum class ContextOrder {
    /** Most relevant passage first, descending. */
    RELEVANCE,

    /**
     * Best passage first, second-best LAST, alternating inward.
     *
     * Long-context models attend most reliably to the beginning and end of their
     * input and are measurably weaker in the middle. When five passages are
     * packed in strict relevance order, the second- and third-best land exactly
     * in the weak zone. This spreads the strongest evidence across both edges.
     */
    EDGES,
}

/**
 * Packs retrieved chunks into a token budget for the prompt.
 *
 * The important job here is overlap removal. The chunker deliberately overlaps
 * consecutive chunks so a sentence split across a boundary survives in at least
 * one of them — but when neighbouring chunks are then concatenated into a
 * prompt, that shared text is duplicated verbatim. Measured against real chunker
 * output, roughly 48% of the assembled characters were duplicates. That is
 * budget spent on text the model has already read, and repeated sentences also
 * skew what it treats as important.
 */
class ContextAssembler(
    private val tokenBudget: Int = 1200,
    private val estimator: TokenEstimator = HeuristicTokenEstimator(),
    private val order: ContextOrder = ContextOrder.EDGES,
    /** Passages this small after trimming carry no information worth a slot. */
    private val minPassageTokens: Int = 12,
    private val injectionPolicy: InjectionPolicy = InjectionPolicy.WITHHOLD,
) {
    init {
        require(tokenBudget > 0) { "tokenBudget must be positive" }
    }

    fun assemble(results: List<RetrievedChunk>): AssembledContext {
        if (results.isEmpty()) return AssembledContext(emptyList(), 0, 0)

        val unique = results.distinctBy { it.chunk.id }
        val withheld = if (injectionPolicy == InjectionPolicy.WITHHOLD) {
            unique.filter { PromptSafety.looksLikeInjection(it.chunk.text) }
        } else {
            emptyList()
        }
        val ranked = unique - withheld.toSet()
        val trimmed = trimOverlaps(ranked)

        var used = 0
        var dropped = ranked.size - trimmed.size
        val selected = ArrayList<Pair<RetrievedChunk, String>>()

        // Selection happens in relevance order regardless of final layout —
        // what makes the cut is decided by score, only the arrangement changes.
        for (candidate in ranked) {
            val text = trimmed[candidate.chunk.id] ?: continue
            val tokens = estimator.estimate(text)
            if (tokens < minPassageTokens) {
                dropped++
                continue
            }
            if (used + tokens > tokenBudget) {
                dropped++
                continue // keep scanning: a later, shorter passage may still fit
            }
            used += tokens
            selected.add(candidate to text)
        }

        val arranged = when (order) {
            ContextOrder.RELEVANCE -> selected
            ContextOrder.EDGES -> arrangeForEdges(selected)
        }

        val passages = arranged.mapIndexed { index, (source, text) ->
            Passage(
                ordinal = index + 1,
                source = source,
                text = text,
                tokens = estimator.estimate(text),
            )
        }
        return AssembledContext(passages, used, dropped, withheld)
    }

    /**
     * Removes text that an earlier-positioned chunk of the same document already
     * covers.
     *
     * Works on character offsets, not string matching: offsets are exact, survive
     * repeated text, and cost nothing to compare. A chunk fully contained in
     * another disappears entirely. Chunks are swept in document order so the
     * result does not depend on the relevance ranking they arrived in.
     */
    private fun trimOverlaps(ranked: List<RetrievedChunk>): Map<String, String> {
        val out = HashMap<String, String>(ranked.size)

        ranked.groupBy { it.chunk.docId }.forEach { (_, inDocument) ->
            var covered = Int.MIN_VALUE
            inDocument.sortedBy { it.chunk.charStart }.forEach { candidate ->
                val chunk = candidate.chunk
                val start = maxOf(chunk.charStart, covered)
                if (start >= chunk.charEnd) return@forEach // wholly redundant

                val offset = start - chunk.charStart
                val text = if (offset > 0) chunk.text.substring(offset).trimStart() else chunk.text
                covered = maxOf(covered, chunk.charEnd)
                if (text.isNotBlank()) out[chunk.id] = text
            }
        }
        return out
    }

    /** [best, 3rd, 5th, ... 6th, 4th, 2nd] — strongest evidence at both edges. */
    private fun <T> arrangeForEdges(ranked: List<T>): List<T> {
        val front = ArrayList<T>(ranked.size)
        val back = ArrayList<T>(ranked.size)
        ranked.forEachIndexed { index, item ->
            if (index % 2 == 0) front.add(item) else back.add(item)
        }
        return front + back.asReversed()
    }
}
