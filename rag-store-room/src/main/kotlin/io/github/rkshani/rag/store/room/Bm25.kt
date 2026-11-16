package io.github.rkshani.rag.store.room

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln

/**
 * BM25 ranking over SQLite FTS4 `matchinfo` output.
 *
 * FTS4 has no built-in `bm25()` (that arrived with FTS5, which Room does not
 * generate and which is not guaranteed present across every Android SQLite
 * build). The supported route is to ask FTS4 for raw statistics and rank in
 * application code.
 *
 * Layout for format string `pcnalx`, confirmed empirically against SQLite rather
 * than taken from documentation — every value is a 32-bit little-endian int:
 *
 *   [0]            p — number of phrases in the query
 *   [1]            c — number of columns in the table
 *   [2]            n — total rows in the table
 *   [3 ..]         a — average length in tokens, one per column
 *   [3+c ..]       l — length of THIS row, one per column
 *   [3+2c ..]      x — 3 ints per (phrase, column):
 *                        hits in this row, hits across all rows, rows containing it
 */
object Bm25 {

    const val MATCHINFO_FORMAT: String = "pcnalx"

    const val DEFAULT_K1: Double = 1.2
    const val DEFAULT_B: Double = 0.75

    /**
     * @param k1 term-frequency saturation. Higher means repeated terms keep adding score.
     * @param b length normalisation, 0..1. At 0.75 a long chunk is penalised moderately.
     */
    fun score(matchInfo: ByteArray, k1: Double = DEFAULT_K1, b: Double = DEFAULT_B): Double {
        require(matchInfo.size % 4 == 0) { "matchinfo blob length must be a multiple of 4" }
        val ints = ByteBuffer.wrap(matchInfo).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer()
        if (ints.limit() < 3) return 0.0

        val phrases = ints.get(0)
        val columns = ints.get(1)
        val totalRows = ints.get(2)
        if (phrases <= 0 || columns <= 0 || totalRows <= 0) return 0.0

        val avgOffset = 3
        val lenOffset = avgOffset + columns
        val statsOffset = lenOffset + columns
        val required = statsOffset + 3 * phrases * columns
        require(ints.limit() >= required) {
            "matchinfo blob is ${ints.limit()} ints, expected at least $required — wrong format string?"
        }

        var score = 0.0
        for (phrase in 0 until phrases) {
            for (column in 0 until columns) {
                val base = statsOffset + 3 * (phrase * columns + column)
                val termFrequency = ints.get(base)
                val rowsWithTerm = ints.get(base + 2)
                if (termFrequency == 0 || rowsWithTerm == 0) continue

                // Probabilistic IDF, +1 smoothed so a term present in most rows
                // scores near zero instead of going negative.
                val idf = ln(1.0 + (totalRows - rowsWithTerm + 0.5) / (rowsWithTerm + 0.5))

                val docLength = ints.get(lenOffset + column).toDouble()
                val averageLength = ints.get(avgOffset + column).toDouble().takeIf { it > 0.0 } ?: 1.0

                val denominator = termFrequency + k1 * (1.0 - b + b * docLength / averageLength)
                score += idf * (termFrequency * (k1 + 1.0)) / denominator
            }
        }
        return score
    }

    /**
     * BM25 scores are unbounded, while vector similarity is 0..1. Phase 4 fuses
     * the two with Reciprocal Rank Fusion, which uses ranks and therefore needs
     * no calibration — but callers that want a comparable number can use this.
     */
    fun normalize(score: Double, maxScore: Double): Float =
        if (maxScore <= 0.0) 0f else (score / maxScore).toFloat()
}
