package io.github.rkshani.rag.core.answer

import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.retrieval.RetrievedChunk

/** A retrieved chunk prepared for the prompt: overlap-trimmed and numbered. */
data class Passage(
    val ordinal: Int,
    val source: RetrievedChunk,
    /** Chunk text with any overlap against an earlier passage removed. */
    val text: String,
    val tokens: Int,
) {
    val chunk: Chunk get() = source.chunk

    /** What the model is told the passage is, and what the UI shows as a citation. */
    fun label(): String = buildString {
        append(chunk.metadata["title"] ?: chunk.docId)
        chunk.page?.let { append(", page ").append(it) }
    }
}

data class AssembledContext(
    val passages: List<Passage>,
    val usedTokens: Int,
    /** Retrieved chunks that did not fit the budget or were fully redundant. */
    val droppedCount: Int,
    /**
     * Retrieved chunks kept out of the prompt because their text reads like
     * instructions aimed at the model. Surface the count — a silently shorter
     * answer is worse than "1 passage was withheld".
     */
    val withheld: List<RetrievedChunk> = emptyList(),
) {
    val isEmpty: Boolean get() = passages.isEmpty()
}

enum class AnswerMode {
    /** The on-device model produced the text. */
    GENERATED,

    /** No usable generator, so the retrieved passages are returned verbatim. */
    EXTRACTIVE,

    /** Retrieval found nothing relevant; no answer was attempted. */
    NO_CONTEXT,
}

data class Answer(
    val text: String,
    val mode: AnswerMode,
    /** Passages the answer actually cited, in citation order. */
    val citations: List<Passage>,
    /** Everything that was put in front of the model. */
    val context: AssembledContext,
    /**
     * Citation markers pointing at passages that were never supplied. A non-empty
     * list means the model invented a source, which is worth surfacing rather
     * than quietly dropping.
     */
    val invalidCitations: List<Int> = emptyList(),
    /** True when generation failed partway; [text] holds what arrived before it. */
    val interrupted: Boolean = false,
    /** The text actually sent to retrieval, after follow-up rewriting. */
    val retrievalQuery: String = "",
)

sealed interface AnswerEvent {
    /** Emitted once retrieval and assembly finish, before any token arrives. */
    data class Context(val context: AssembledContext, val retrievalQuery: String = "") : AnswerEvent

    /** A partial chunk of generated text. */
    data class Token(val text: String) : AnswerEvent

    data class Completed(val answer: Answer) : AnswerEvent
}
