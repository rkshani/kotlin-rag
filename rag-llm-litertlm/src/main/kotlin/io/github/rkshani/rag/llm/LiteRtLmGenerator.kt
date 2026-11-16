package io.github.rkshani.rag.llm

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import io.github.rkshani.rag.core.GenerationOptions
import io.github.rkshani.rag.core.Generator
import io.github.rkshani.rag.core.RagDispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * [Generator] on LiteRT-LM, Google's current on-device LLM runtime and the
 * successor to the MediaPipe LLM Inference task this module used before.
 *
 * Written against the LiteRT-LM Kotlin API as documented at
 * https://developers.google.com/edge/litert-lm/android: `Engine(EngineConfig)`,
 * `initialize()`, one `Conversation` per answer, `sendMessageAsync(String)`
 * returning `Flow<Message>`, `cancelProcess()`. If a newer release renames any
 * of these, this file is the only one to touch.
 *
 * - The engine is created lazily on first use, on the IO dispatcher:
 *   `initialize()` can take ~10 s and allocates over a gigabyte.
 * - Each answer gets a fresh conversation. RAG prompts already carry their own
 *   history, and a long-lived conversation would grow its KV cache forever.
 * - LiteRT-LM applies the model's chat template itself, so prompts arrive
 *   untemplated ([appliesChatTemplate] = true).
 * - One generation at a time; a second caller waits.
 */
class LiteRtLmGenerator(
    private val context: Context,
    private val modelPath: String,
    private val backend: () -> Backend = { Backend.CPU() },
    private val dispatchers: RagDispatchers = RagDispatchers(),
    /** Called with the reason when the device check rejects the model. */
    private val onUnavailable: (String) -> Unit = {},
) : Generator {

    private val engineLock = Mutex()
    private val generationLock = Mutex()
    private var engine: Engine? = null

    override val appliesChatTemplate: Boolean = true

    override suspend fun isAvailable(): Boolean = withContext(dispatchers.io) {
        val file = File(modelPath)
        if (!file.isFile || file.length() == 0L) return@withContext false
        val reason = DeviceCapability.inspect(context).reason(file.length())
        if (reason != null) onUnavailable(reason)
        reason == null
    }

    override fun generate(prompt: String, options: GenerationOptions): Flow<String> = flow {
        val engine = engine()
        generationLock.withLock {
            val conversation = engine.createConversation(
                ConversationConfig(
                    samplerConfig = SamplerConfig(
                        // temperature 0 is not a valid sampling temperature; greedy
                        // decoding is topK = 1 instead.
                        topK = if (options.temperature <= 0f) 1 else options.topK,
                        topP = 0.95,
                        temperature = options.temperature.coerceAtLeast(0.01f).toDouble(),
                    ),
                ),
            )
            // The engine's token limit is fixed per engine, and its streamed
            // messages are not single tokens, so the per-answer budget is
            // enforced here as a character budget (~4 chars per token).
            val charBudget = options.maxTokens.toLong() * 4
            var produced = 0L
            try {
                conversation.sendMessageAsync(prompt)
                    .takeWhile { produced < charBudget }
                    .collect { message ->
                        val text = message.toString()
                        produced += text.length
                        emit(text)
                    }
                // takeWhile stops collecting but not the native decoder.
                if (produced >= charBudget) runCatching { conversation.cancelProcess() }
            } catch (cancelled: CancellationException) {
                // Stop native decoding too — otherwise it keeps burning CPU and
                // battery after the user has left the screen.
                runCatching { conversation.cancelProcess() }
                throw cancelled
            } finally {
                runCatching { conversation.close() }
            }
        }
    }.flowOn(dispatchers.io)

    private suspend fun engine(): Engine = engineLock.withLock {
        engine ?: withContext(dispatchers.io) {
            check(File(modelPath).isFile) { "model file missing: $modelPath" }
            Engine(
                EngineConfig(
                    modelPath = modelPath,
                    backend = backend(),
                    cacheDir = context.cacheDir.path,
                ),
            ).also { it.initialize() }
        }.also { engine = it }
    }

    override fun close() {
        runCatching { engine?.close() }
        engine = null
    }
}
