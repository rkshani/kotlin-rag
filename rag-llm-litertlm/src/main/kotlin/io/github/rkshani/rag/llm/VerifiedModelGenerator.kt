package io.github.rkshani.rag.llm

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import io.github.rkshani.rag.core.GenerationOptions
import io.github.rkshani.rag.core.Generator
import io.github.rkshani.rag.core.RagDispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * A [Generator] for a model that may not be downloaded yet.
 *
 * Reports unavailable until [ModelDownloader] has fetched and verified the file,
 * then starts serving — so the app can be built once at startup and answers
 * switch from extractive to generated the moment the download finishes. Only a
 * file carrying the downloader's verification marker is ever loaded.
 */
class VerifiedModelGenerator(
    private val context: Context,
    private val spec: ModelSpec,
    private val backend: () -> Backend = { Backend.CPU() },
    private val dispatchers: RagDispatchers = RagDispatchers(),
    private val onUnavailable: (String) -> Unit = {},
) : Generator {

    private val lock = Mutex()
    private var delegate: LiteRtLmGenerator? = null

    override val appliesChatTemplate: Boolean = true

    override suspend fun isAvailable(): Boolean = current()?.isAvailable() ?: false

    override fun generate(prompt: String, options: GenerationOptions): Flow<String> = flow {
        val generator = checkNotNull(current()) { "model ${spec.fileName} is not downloaded" }
        emitAll(generator.generate(prompt, options))
    }

    private suspend fun current(): LiteRtLmGenerator? = lock.withLock {
        val file = ModelDownloader.verifiedFile(context, spec)
        if (file == null) {
            // Deleted or replaced since it was loaded: drop the engine with it.
            delegate?.close()
            delegate = null
            return@withLock null
        }
        delegate ?: LiteRtLmGenerator(context, file.absolutePath, backend, dispatchers, onUnavailable).also { delegate = it }
    }

    override fun close() {
        delegate?.close()
        delegate = null
    }
}
