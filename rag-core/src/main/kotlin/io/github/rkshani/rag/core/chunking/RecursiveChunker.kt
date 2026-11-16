package io.github.rkshani.rag.core.chunking

import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.SourceDocument
import kotlin.math.max

/**
 * Splits on the largest natural boundary that fits, falling back through
 * paragraph -> line -> sentence -> word -> hard cut, then merges neighbouring
 * pieces back up to [targetTokens] with [overlapTokens] of carry-over.
 *
 * Every emitted chunk carries offsets into the original text, so nothing is
 * re-derived by string search later (which breaks on repeated text).
 */
class RecursiveChunker(
    private val targetTokens: Int = 400,
    private val overlapTokens: Int = 50,
    private val minTokens: Int = 40,
    private val tokenEstimator: TokenEstimator = HeuristicTokenEstimator(),
    private val separators: List<String> = DEFAULT_SEPARATORS,
) : Chunker {

    init {
        require(targetTokens > 0) { "targetTokens must be positive" }
        require(overlapTokens in 0 until targetTokens) { "overlapTokens must be in 0 until targetTokens" }
    }

    private data class Span(val start: Int, val end: Int) {
        val length get() = end - start
    }

    override fun chunk(document: SourceDocument): List<Chunk> {
        val text = document.text
        if (text.isBlank()) return emptyList()

        val atoms = split(text, Span(0, text.length), sepIndex = 0)
        val merged = merge(text, atoms)

        return merged.mapIndexedNotNull { i, span ->
            val trimmed = trim(text, span) ?: return@mapIndexedNotNull null
            Chunk(
                id = "${document.id}#$i",
                docId = document.id,
                index = i,
                text = text.substring(trimmed.start, trimmed.end),
                charStart = trimmed.start,
                charEnd = trimmed.end,
                page = document.pageAt(trimmed.start),
                metadata = document.metadata,
            )
        }
    }

    /** Recursively break [span] down until every piece fits in [targetTokens]. */
    private fun split(text: String, span: Span, sepIndex: Int): List<Span> {
        if (span.length == 0) return emptyList()
        if (tokensOf(text, span) <= targetTokens) return listOf(span)
        if (sepIndex >= separators.size) return hardSplit(text, span)

        val parts = splitOn(text, span, separators[sepIndex])
        // Separator absent (or produced a single piece): try the next one down.
        if (parts.size <= 1) return split(text, span, sepIndex + 1)

        return parts.flatMap { split(text, it, sepIndex + 1) }
    }

    /**
     * Splits on [separator], keeping the separator attached to the piece that
     * precedes it so that concatenating all spans reproduces the input exactly.
     */
    private fun splitOn(text: String, span: Span, separator: String): List<Span> {
        val out = mutableListOf<Span>()
        var cursor = span.start
        while (cursor < span.end) {
            val hit = text.indexOf(separator, cursor)
            if (hit < 0 || hit + separator.length > span.end) break
            out += Span(cursor, hit + separator.length)
            cursor = hit + separator.length
        }
        if (cursor < span.end) out += Span(cursor, span.end)
        return out.filter { it.length > 0 }
    }

    /**
     * Last resort for pathological input (a 10k-character run with no spaces —
     * base64 blobs, minified JS, some OCR output). Window size is derived from
     * the estimator so it adapts to script rather than assuming 4 chars/token.
     */
    private fun hardSplit(text: String, span: Span): List<Span> {
        val tokens = max(1, tokensOf(text, span))
        val charsPerToken = span.length.toDouble() / tokens
        val window = max(1, (targetTokens * charsPerToken).toInt())
        val out = mutableListOf<Span>()
        var cursor = span.start
        while (cursor < span.end) {
            val end = minOf(cursor + window, span.end)
            out += Span(cursor, end)
            cursor = end
        }
        return out
    }

    /** Greedily packs atoms up to [targetTokens], then rewinds for overlap. */
    private fun merge(text: String, atoms: List<Span>): List<Span> {
        if (atoms.isEmpty()) return emptyList()
        val sizes = IntArray(atoms.size) { tokensOf(text, atoms[it]) }
        val out = mutableListOf<Span>()

        var i = 0
        while (i < atoms.size) {
            var end = i
            var budget = 0
            while (end < atoms.size) {
                val size = sizes[end]
                if (budget > 0 && budget + size > targetTokens) break
                budget += size
                end++
            }
            out += Span(atoms[i].start, atoms[end - 1].end)
            if (end >= atoms.size) break

            // Rewind so the next chunk repeats ~overlapTokens of context.
            var back = end
            var carried = 0
            while (back > i + 1 && carried < overlapTokens) {
                back--
                carried += sizes[back]
            }
            i = if (back <= i) i + 1 else back
        }

        return foldTinyTail(text, out)
    }

    /**
     * A trailing fragment shorter than [minTokens] retrieves badly and pollutes
     * results, so it gets absorbed into the previous chunk instead.
     */
    private fun foldTinyTail(text: String, spans: List<Span>): List<Span> {
        if (spans.size < 2) return spans
        val last = spans.last()
        if (tokensOf(text, last) >= minTokens) return spans
        val head = spans.dropLast(1).toMutableList()
        val prev = head.removeAt(head.lastIndex)
        val merged = Span(prev.start, maxOf(prev.end, last.end))
        // Folding must not break the budget: with a small target (a 256-token
        // embedder window) a "tiny" tail can be most of a chunk, and the merged
        // chunk would be silently truncated by the embedder.
        if (tokensOf(text, merged) > targetTokens) return spans
        head += merged
        return head
    }

    private fun trim(text: String, span: Span): Span? {
        var start = span.start
        var end = span.end
        while (start < end && text[start].isWhitespace()) start++
        while (end > start && text[end - 1].isWhitespace()) end--
        return if (start >= end) null else Span(start, end)
    }

    private fun tokensOf(text: String, span: Span): Int =
        tokenEstimator.estimate(text.substring(span.start, span.end))

    companion object {
        /** Ordered widest-to-narrowest. Urdu full stop (۔) sits alongside the Latin one. */
        val DEFAULT_SEPARATORS: List<String> = listOf(
            "\n\n", "\n", "۔ ", "। ", ". ", "! ", "? ", "; ", ", ", " ",
        )
    }
}
