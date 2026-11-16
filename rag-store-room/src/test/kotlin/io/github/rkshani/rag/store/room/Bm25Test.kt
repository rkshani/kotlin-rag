package io.github.rkshani.rag.store.room

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The blobs below are real `matchinfo(t, 'pcnalx')` output captured from SQLite
 * FTS4 over a four-row table, and the expected scores come from an independent
 * reference implementation of the same formula. If these fail, the byte layout
 * assumption in [Bm25] is wrong.
 */
class Bm25Test {

    private fun matchInfo(vararg values: Int): ByteArray {
        val bytes = ByteArray(values.size * 4)
        values.forEachIndexed { i, value ->
            val o = i * 4
            bytes[o] = (value and 0xFF).toByte()
            bytes[o + 1] = ((value ushr 8) and 0xFF).toByte()
            bytes[o + 2] = ((value ushr 16) and 0xFF).toByte()
            bytes[o + 3] = ((value ushr 24) and 0xFF).toByte()
        }
        return bytes
    }

    // 1 phrase, 1 column, 4 rows, avg len 7, this row 9 tokens, term appears
    // once here / 4 times overall / in 2 rows.
    private val singleTermLongRow = matchInfo(1, 1, 4, 7, 9, 1, 4, 2)

    // Same query, a row where the term appears three times in 8 tokens.
    private val singleTermRepeated = matchInfo(1, 1, 4, 7, 8, 3, 4, 2)

    // Two-phrase query "quick dog" against the first row.
    private val twoPhrases = matchInfo(2, 1, 4, 7, 9, 1, 4, 2, 1, 2, 2)

    @Test
    fun `scores match the reference implementation`() {
        assertEquals(0.620609, Bm25.score(singleTermLongRow), 1e-5)
        assertEquals(1.056878, Bm25.score(singleTermRepeated), 1e-5)
        assertEquals(1.241217, Bm25.score(twoPhrases), 1e-5)
    }

    @Test
    fun `repeated terms in a shorter row outrank a single hit in a longer row`() {
        assertTrue(Bm25.score(singleTermRepeated) > Bm25.score(singleTermLongRow))
    }

    @Test
    fun `matching more query terms scores higher`() {
        assertTrue(Bm25.score(twoPhrases) > Bm25.score(singleTermLongRow))
    }

    @Test
    fun `term frequency saturates rather than growing linearly`() {
        val onceIn10 = matchInfo(1, 1, 100, 10, 10, 1, 50, 10)
        val fiveTimesIn10 = matchInfo(1, 1, 100, 10, 10, 5, 50, 10)
        val tenTimesIn10 = matchInfo(1, 1, 100, 10, 10, 10, 50, 10)

        val first = Bm25.score(fiveTimesIn10) - Bm25.score(onceIn10)
        val second = Bm25.score(tenTimesIn10) - Bm25.score(fiveTimesIn10)
        assertTrue(second < first, "k1 saturation is not being applied")
    }

    @Test
    fun `a term present in nearly every row contributes almost nothing`() {
        val rare = matchInfo(1, 1, 1000, 10, 10, 1, 5, 5)
        val ubiquitous = matchInfo(1, 1, 1000, 10, 10, 1, 5000, 999)
        assertTrue(Bm25.score(ubiquitous) < Bm25.score(rare) / 10)
    }

    @Test
    fun `longer rows are penalised for the same term frequency`() {
        val shortRow = matchInfo(1, 1, 100, 50, 20, 3, 200, 40)
        val longRow = matchInfo(1, 1, 100, 50, 400, 3, 200, 40)
        assertTrue(Bm25.score(shortRow) > Bm25.score(longRow))
    }

    @Test
    fun `idf never goes negative`() {
        val everyRow = matchInfo(1, 1, 10, 10, 10, 1, 10, 10)
        assertTrue(Bm25.score(everyRow) >= 0.0)
    }

    @Test
    fun `degenerate blobs score zero instead of throwing`() {
        assertEquals(0.0, Bm25.score(matchInfo(0, 0, 0)), 0.0)
        assertEquals(0.0, Bm25.score(ByteArray(0)), 0.0)
        assertEquals(0.0, Bm25.score(matchInfo(1, 1, 4, 7, 9, 0, 0, 0)), 0.0)
    }

    @Test
    fun `zero average length does not divide by zero`() {
        val score = Bm25.score(matchInfo(1, 1, 4, 0, 5, 1, 4, 2))
        assertTrue(score.isFinite())
    }
}

class FtsQueryTest {

    @Test
    fun `terms are quoted and joined with OR`() {
        assertEquals("\"machine\" OR \"learning\"", FtsQuery.sanitize("machine learning"))
    }

    @Test
    fun `fts operators in user input are neutralised`() {
        // Without quoting, these would be parsed as FTS syntax rather than terms.
        assertEquals("\"cats\" OR \"NEAR\" OR \"dogs\"", FtsQuery.sanitize("cats NEAR dogs"))
    }

    @Test
    fun `punctuation and quotes are stripped`() {
        assertEquals("\"don\" OR \"t\" OR \"stop\"", FtsQuery.sanitize("don't \"stop\"!"))
    }

    @Test
    fun `non-latin scripts survive tokenization`() {
        assertEquals("\"اردو\" OR \"متن\"", FtsQuery.sanitize("اردو متن"))
    }

    @Test
    fun `empty or symbol-only input yields null`() {
        assertEquals(null, FtsQuery.sanitize(""))
        assertEquals(null, FtsQuery.sanitize("   "))
        assertEquals(null, FtsQuery.sanitize("!!! ??? ***"))
    }
}

class TopKTest {

    @Test
    fun `keeps the highest scoring items in descending order`() {
        val topK = TopK<Pair<String, Float>>(3) { it.second }
        listOf("a" to 0.1f, "b" to 0.9f, "c" to 0.5f, "d" to 0.7f, "e" to 0.2f).forEach(topK::offer)

        assertEquals(listOf("b", "d", "c"), topK.drain().map { it.first })
    }

    @Test
    fun `returns everything when fewer items than k`() {
        val topK = TopK<Pair<String, Float>>(10) { it.second }
        listOf("a" to 0.1f, "b" to 0.9f).forEach(topK::offer)

        assertEquals(listOf("b", "a"), topK.drain().map { it.first })
    }

    @Test
    fun `handles an empty stream`() {
        assertTrue(TopK<Pair<String, Float>>(5) { it.second }.drain().isEmpty())
    }
}
