package io.github.rkshani.rag.core.model

/**
 * A contiguous slice of a source document, small enough to embed.
 *
 * [charStart] and [charEnd] always refer to offsets in the ORIGINAL document text,
 * which is what lets the UI jump back to the exact spot when citing a source.
 */
data class Chunk(
    val id: String,
    val docId: String,
    val index: Int,
    val text: String,
    val charStart: Int,
    val charEnd: Int,
    val page: Int? = null,
    val metadata: Map<String, String> = emptyMap(),
) {
    init {
        require(charEnd >= charStart) { "charEnd must be >= charStart" }
    }
}

/**
 * A [Chunk] plus its vector. Deliberately not a data class: [FloatArray] has
 * identity equals, so a generated equals() would be misleading.
 */
class EmbeddedChunk(val chunk: Chunk, val embedding: FloatArray) {
    override fun toString(): String = "EmbeddedChunk(${chunk.id}, dim=${embedding.size})"
}

/** A retrieval hit. [score] is higher-is-better regardless of the underlying metric. */
data class ScoredChunk(val chunk: Chunk, val score: Float)

/**
 * Input to the chunker.
 *
 * @param pageOffsets character offset at which each page starts, ascending.
 *   Empty when the source has no page structure (plain text, markdown).
 */
data class SourceDocument(
    val id: String,
    val text: String,
    val title: String? = null,
    val pageOffsets: List<Int> = emptyList(),
    val metadata: Map<String, String> = emptyMap(),
) {
    /** 1-based page number containing [charOffset], or null when unpaged. */
    fun pageAt(charOffset: Int): Int? {
        if (pageOffsets.isEmpty()) return null
        var lo = 0
        var hi = pageOffsets.lastIndex
        var ans = 0
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (pageOffsets[mid] <= charOffset) {
                ans = mid
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        return ans + 1
    }
}
