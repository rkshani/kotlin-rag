package io.github.rkshani.rag.embed.onnx

/**
 * A padded batch ready to feed ONNX Runtime. Flat arrays, row-major,
 * [batchSize] x [sequenceLength].
 */
class TokenBatch(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val tokenTypeIds: LongArray,
    val batchSize: Int,
    val sequenceLength: Int,
) {
    init {
        val expected = batchSize * sequenceLength
        require(inputIds.size == expected) { "inputIds size ${inputIds.size} != $expected" }
        require(attentionMask.size == expected) { "attentionMask size ${attentionMask.size} != $expected" }
        require(tokenTypeIds.size == expected) { "tokenTypeIds size ${tokenTypeIds.size} != $expected" }
    }

    companion object {
        val EMPTY = TokenBatch(LongArray(0), LongArray(0), LongArray(0), 0, 0)
    }
}

interface Tokenizer {
    /** Token ids for one text, WITHOUT special tokens or padding. */
    fun encode(text: String): IntArray

    /** Human-readable pieces. Used by the parity test and for debugging only. */
    fun tokenize(text: String): List<String>

    /**
     * Encodes a batch: adds the model's special tokens, truncates to
     * [maxSequenceLength], pads to the longest row in the batch.
     */
    fun encodeBatch(texts: List<String>, maxSequenceLength: Int): TokenBatch

    /**
     * Encodes (first, second[i]) pairs for a cross-encoder: one row per
     * [seconds] entry, each holding both texts. When a pair is too long the
     * longer side is cut first, so a short query is never truncated away.
     */
    fun encodePairBatch(first: String, seconds: List<String>, maxSequenceLength: Int): TokenBatch

    val vocabSize: Int
}

/** Cuts the longer of two id sequences until both fit in [budget] tokens together. */
internal fun truncatePair(a: IntArray, b: IntArray, budget: Int): Pair<IntArray, IntArray> {
    var lenA = a.size
    var lenB = b.size
    while (lenA + lenB > budget) {
        if (lenB >= lenA && lenB > 0) lenB-- else if (lenA > 0) lenA-- else break
    }
    return a.copyOf(lenA) to b.copyOf(lenB)
}

/**
 * Pads rows to the longest one in THIS batch, not to the model maximum —
 * padding to 256 when the batch needs 40 wastes ~6x the compute.
 */
internal fun padBatch(rows: List<IntArray>, types: List<IntArray>?, padId: Int): TokenBatch {
    if (rows.isEmpty()) return TokenBatch.EMPTY
    val seqLen = rows.maxOf { it.size }
    val total = rows.size * seqLen
    val inputIds = LongArray(total) { padId.toLong() }
    val mask = LongArray(total)
    val tokenTypes = LongArray(total)

    rows.forEachIndexed { r, row ->
        val base = r * seqLen
        for (c in row.indices) {
            inputIds[base + c] = row[c].toLong()
            mask[base + c] = 1L
            if (types != null) tokenTypes[base + c] = types[r][c].toLong()
        }
    }
    return TokenBatch(inputIds, mask, tokenTypes, rows.size, seqLen)
}
