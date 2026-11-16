package io.github.rkshani.rag.core.answer

import io.github.rkshani.rag.core.GenerationOptions
import io.github.rkshani.rag.core.Generator
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.chat.ChatTurn
import io.github.rkshani.rag.core.chat.QueryRewriter
import io.github.rkshani.rag.core.retrieval.Retriever
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * End-to-end question answering: rewrite, retrieve, assemble, generate, cite.
 *
 * The generator is optional and may be unavailable at runtime. That is not an
 * edge case on device — the model is a ~1 GB download that may not have finished,
 * may have been evicted, or may simply not fit the hardware. When it is missing,
 * this returns the retrieved passages as an extractive answer instead of failing.
 * Search that works without the LLM is far more useful than an error screen.
 */
class RagAnswerer(
    private val retriever: Retriever,
    private val generator: Generator? = null,
    private val assembler: ContextAssembler = ContextAssembler(),
    private val promptBuilder: PromptBuilder = PromptBuilder(),
    private val config: AnswerConfig = AnswerConfig(),
    private val queryRewriter: QueryRewriter = QueryRewriter.NONE,
    /** Called when generation fails; the answer still completes via a fallback. */
    private val onGenerationFailure: (Throwable) -> Unit = {},
) {

    data class AnswerConfig(
        val retrieveK: Int = 5,
        val generation: GenerationOptions = GenerationOptions(),
        /** Passages included in an extractive answer when generation is unavailable. */
        val extractivePassages: Int = 3,
        val noContextMessage: String =
            "I could not find anything about that in your documents.",
    )

    /**
     * Streams the answer. Emits [AnswerEvent.Context] first so the UI can show
     * sources immediately — retrieval takes milliseconds while generation takes
     * seconds, and showing what was found makes the wait legible.
     */
    fun answer(
        question: String,
        filter: SearchFilter? = null,
        history: List<ChatTurn> = emptyList(),
    ): Flow<AnswerEvent> = flow {
        val asked = question.trim().take(promptBuilder.maxQuestionChars)
        val retrievalQuery = if (history.isEmpty()) asked else queryRewriter.rewrite(asked, history)
        val retrieved = retriever.retrieve(retrievalQuery, config.retrieveK, filter)
        val context = assembler.assemble(retrieved)
        emit(AnswerEvent.Context(context, retrievalQuery))

        if (context.isEmpty) {
            emit(
                AnswerEvent.Completed(
                    Answer(
                        text = config.noContextMessage,
                        mode = AnswerMode.NO_CONTEXT,
                        citations = emptyList(),
                        context = context,
                        retrievalQuery = retrievalQuery,
                    ),
                ),
            )
            return@flow
        }

        val usable = generator?.takeIf { isAvailable(it) }
        if (usable == null) {
            emit(AnswerEvent.Completed(extractiveAnswer(context, retrievalQuery)))
            return@flow
        }

        val prompt = promptBuilder.build(asked, context, history, applyTemplate = !usable.appliesChatTemplate)
        val builder = StringBuilder()
        var failed = false
        try {
            usable.generate(prompt, config.generation).collect { token ->
                builder.append(token)
                emit(AnswerEvent.Token(token))
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            // Native engines fail for mundane reasons — memory pressure, a model
            // evicted mid-run. The user still gets an answer.
            onGenerationFailure(failure)
            failed = true
        }

        val text = builder.toString().trim()
        if (text.isEmpty()) {
            emit(AnswerEvent.Completed(extractiveAnswer(context, retrievalQuery)))
            return@flow
        }

        val parsed = Citations.parse(text, context.passages.size)
        emit(
            AnswerEvent.Completed(
                Answer(
                    text = text,
                    mode = AnswerMode.GENERATED,
                    citations = Citations.resolve(parsed, context.passages),
                    context = context,
                    invalidCitations = parsed.invalid,
                    interrupted = failed,
                    retrievalQuery = retrievalQuery,
                ),
            ),
        )
    }

    private suspend fun isAvailable(generator: Generator): Boolean = try {
        generator.isAvailable()
    } catch (cancelled: CancellationException) {
        throw cancelled
    } catch (failure: Throwable) {
        false
    }

    private fun extractiveAnswer(context: AssembledContext, retrievalQuery: String): Answer {
        // Present in relevance order, not the edge arrangement used for prompts:
        // a human reads top-down and wants the best match first.
        val best = context.passages
            .sortedByDescending { it.source.score }
            .take(config.extractivePassages)

        val text = buildString {
            appendLine("Here is what your documents say:")
            appendLine()
            best.forEach { passage ->
                appendLine("• ${passage.text.trim()}")
                appendLine("  — ${passage.label()}")
                appendLine()
            }
        }.trim()

        return Answer(
            text = text,
            mode = AnswerMode.EXTRACTIVE,
            citations = best,
            context = context,
            retrievalQuery = retrievalQuery,
        )
    }
}
