package io.github.rkshani.rag.core.chunking

import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.SourceDocument
import kotlin.math.max
import kotlin.math.roundToInt

interface Chunker {
    fun chunk(document: SourceDocument): List<Chunk>
}

/**
 * Estimates token count for a string.
 *
 * Phase 1 ships a heuristic so `rag-core` stays dependency-free and JVM-testable.
 * In Phase 2 the real WordPiece tokenizer is injected here, and chunk sizes become
 * exact instead of approximate.
 */
fun interface TokenEstimator {
    fun estimate(text: String): Int
}

/**
 * Character-ratio heuristic.
 *
 * Latin text runs about 3.8 characters per WordPiece token. Arabic-script text
 * (Urdu, Persian, Arabic) fragments much harder under a multilingual vocab —
 * closer to 2.2 — so a fixed chars/4 rule badly undercounts Urdu and produces
 * chunks that silently overflow the model's 256/512-token window.
 */
class HeuristicTokenEstimator(
    private val latinCharsPerToken: Double = 3.8,
    private val arabicCharsPerToken: Double = 2.2,
    private val cjkCharsPerToken: Double = 1.1,
) : TokenEstimator {

    override fun estimate(text: String): Int {
        if (text.isBlank()) return 0
        var arabic = 0
        var cjk = 0
        var other = 0
        for (ch in text) {
            if (ch.isWhitespace()) continue
            when {
                isArabicScript(ch) -> arabic++
                isCjk(ch) -> cjk++
                else -> other++
            }
        }
        val total = arabic + cjk + other
        if (total == 0) return 0
        val tokens = arabic / arabicCharsPerToken + cjk / cjkCharsPerToken + other / latinCharsPerToken
        return max(1, tokens.roundToInt())
    }

    private fun isArabicScript(ch: Char): Boolean {
        val c = ch.code
        // Arabic, Arabic Supplement, Arabic Extended-A, Arabic Presentation Forms A/B
        return c in 0x0600..0x06FF || c in 0x0750..0x077F || c in 0x08A0..0x08FF ||
            c in 0xFB50..0xFDFF || c in 0xFE70..0xFEFF
    }

    private fun isCjk(ch: Char): Boolean {
        val c = ch.code
        return c in 0x4E00..0x9FFF || c in 0x3040..0x30FF || c in 0xAC00..0xD7AF
    }
}
