package io.github.rkshani.rag.core.chat

import io.github.rkshani.rag.core.GenerationOptions
import io.github.rkshani.rag.core.Generator
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.answer.AnswerEvent
import io.github.rkshani.rag.core.answer.AnswerMode
import io.github.rkshani.rag.core.answer.PromptSafety
import io.github.rkshani.rag.core.answer.PromptTemplate
import io.github.rkshani.rag.core.answer.RagAnswerer
import io.github.rkshani.rag.core.text.SearchText
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.fold
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull

data class ChatTurn(val question: String, val answer: String)

/**
 * Turns a follow-up question into something retrieval can use on its own.
 *
 * "Why is it slow?" means nothing to a search index — "it" is in the previous
 * turn. The rewrite only feeds retrieval; the model still sees the question as
 * the user typed it, together with the history.
 */
fun interface QueryRewriter {
    suspend fun rewrite(question: String, history: List<ChatTurn>): String

    companion object {
        val NONE: QueryRewriter = QueryRewriter { question, _ -> question }
    }
}

/**
 * Cheap, model-free rewriting: when a question looks like a follow-up, the
 * previous question is prepended. Hybrid retrieval copes well with the longer
 * query — BM25 picks up the subject's keywords and the embedding averages the
 * two intents.
 */
class FollowUpQueryRewriter(
    private val maxWordsForFollowUp: Int = 6,
    private val maxQueryChars: Int = 400,
) : QueryRewriter {

    override suspend fun rewrite(question: String, history: List<ChatTurn>): String {
        val previous = history.lastOrNull()?.question ?: return question
        if (!isFollowUp(question)) return question
        val combined = "$previous $question"
        return if (combined.length <= maxQueryChars) combined else combined.takeLast(maxQueryChars)
    }

    fun isFollowUp(question: String): Boolean {
        val words = SearchText.normalize(question)
            .split(Regex("""[^\p{L}\p{N}']+"""))
            .filter { it.isNotBlank() }
        if (words.isEmpty()) return false
        if (words.size <= maxWordsForFollowUp && words.size < 3) return true
        if (words.first() in LEADING_CONNECTIVES) return true
        return words.size <= maxWordsForFollowUp && words.any { it in REFERENCES }
    }

    private companion object {
        val REFERENCES = setOf(
            "it", "its", "this", "that", "these", "those", "they", "them", "their", "he", "she", "him", "her",
            "there", "same", "above", "previous",
            // Roman Urdu
            "ye", "yeh", "wo", "woh", "is", "us", "isko", "usko", "iska", "uska", "iski", "uski", "inka", "unka",
            // Urdu (after SearchText folding)
            "یہ", "وہ", "اس", "ان", "اسے", "انہیں", "اسکا", "اسکی", "انکا",
        )
        val LEADING_CONNECTIVES = setOf("and", "also", "but", "so", "what about", "aur", "phir", "اور", "پھر")
    }
}

/**
 * Asks the generator to condense the conversation into a standalone query.
 *
 * Better than [FollowUpQueryRewriter] on long, winding conversations, but it is
 * an extra full model call before retrieval even starts — seconds on a phone.
 * Bounded by [timeoutMillis], and every failure falls back to [fallback].
 */
class LlmQueryRewriter(
    private val generator: Generator,
    private val template: PromptTemplate = PromptTemplate.Gemma,
    private val fallback: QueryRewriter = FollowUpQueryRewriter(),
    private val timeoutMillis: Long = 4_000,
    private val maxQueryChars: Int = 300,
) : QueryRewriter {

    override suspend fun rewrite(question: String, history: List<ChatTurn>): String {
        if (history.isEmpty()) return question
        val available = try {
            generator.isAvailable()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            false
        }
        if (!available) return fallback.rewrite(question, history)

        val conversation = history.takeLast(3).joinToString("\n") { turn ->
            "User: ${PromptSafety.sanitize(turn.question).take(300)}\nAssistant: ${PromptSafety.sanitize(turn.answer).take(300)}"
        }
        val prompt = (if (generator.appliesChatTemplate) PromptTemplate.Plain else template).wrap(
            "Rewrite the user's last question as one standalone search query. " +
                "Resolve words like 'it' or 'that' using the conversation. " +
                "Reply with the query only, on one line, in the language of the question.",
            "$conversation\n\nLast question: ${PromptSafety.sanitize(question).take(500)}\n\nStandalone query:",
        )

        val rewritten = try {
            withTimeoutOrNull(timeoutMillis) {
                generator.generate(prompt, GenerationOptions(maxTokens = 64, temperature = 0f))
                    .fold(StringBuilder()) { acc, token -> acc.append(token) }
                    .toString()
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            null
        }

        val line = rewritten?.lineSequence()?.firstOrNull { it.isNotBlank() }?.trim()?.trim('"', '\'')
        return if (line.isNullOrBlank() || line.length > maxQueryChars) fallback.rewrite(question, history) else line
    }
}

/**
 * A multi-turn conversation over the indexed documents.
 *
 * Holds the last [maxTurns] exchanges, feeds them to query rewriting and to the
 * prompt, and records each completed answer. One question at a time: a second
 * [ask] waits for the first to finish, so turns can never interleave.
 */
class ChatSession(
    private val answerer: RagAnswerer,
    private val maxTurns: Int = 6,
) {
    private val lock = Mutex()
    private val turns = ArrayList<ChatTurn>()

    val history: List<ChatTurn> get() = synchronized(turns) { turns.toList() }

    fun ask(question: String, filter: SearchFilter? = null): Flow<AnswerEvent> = flow {
        lock.withLock {
            answerer.answer(question, filter, history).collect { event ->
                if (event is AnswerEvent.Completed && event.answer.mode != AnswerMode.NO_CONTEXT) {
                    record(ChatTurn(question, event.answer.text))
                }
                emit(event)
            }
        }
    }

    fun clear() = synchronized(turns) { turns.clear() }

    private fun record(turn: ChatTurn) = synchronized(turns) {
        turns.add(turn)
        while (turns.size > maxTurns) turns.removeAt(0)
    }
}
