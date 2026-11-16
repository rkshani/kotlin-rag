package io.github.rkshani.rag.embed.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import io.github.rkshani.rag.core.Embedder
import io.github.rkshani.rag.core.EmbeddingKind
import io.github.rkshani.rag.core.RagDispatchers
import io.github.rkshani.rag.core.chunking.TokenEstimator
import io.github.rkshani.rag.core.math.Vectors
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream
import java.nio.LongBuffer
import kotlin.math.min

/**
 * Sentence embeddings from a transformer exported to ONNX, run entirely on device.
 *
 * Thread-safe: ORT sessions are not, so runs are serialised behind a mutex. One
 * session is created and reused — creating a session per call costs ~200 ms and
 * re-reads the whole model.
 */
class OnnxEmbedder private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    val tokenizer: Tokenizer,
    private val spec: EmbeddingModelSpec,
    private val dispatchers: RagDispatchers,
) : Embedder {

    override val dimensions: Int = spec.dimensions
    override val maxSequenceTokens: Int = spec.maxSequenceTokens

    private val mutex = Mutex()
    private val needsTokenTypeIds = "token_type_ids" in session.inputNames

    /**
     * Exact token counts from the model's own tokenizer, for the chunker. The
     * default heuristic only estimates, and a chunk that is really longer than
     * the window gets its tail cut off unseen.
     */
    fun tokenEstimator(): TokenEstimator {
        val prefixTokens = tokenizer.encode(spec.documentPrefix).size + 2 // + the two special tokens
        return TokenEstimator { text -> tokenizer.encode(text).size + prefixTokens }
    }

    override suspend fun embed(texts: List<String>, kind: EmbeddingKind): List<FloatArray> {
        if (texts.isEmpty()) return emptyList()

        val prefix = if (kind == EmbeddingKind.QUERY) spec.queryPrefix else spec.documentPrefix
        val prepared = if (prefix.isEmpty()) texts else texts.map { prefix + it }

        return withContext(dispatchers.default) {
            val out = arrayOfNulls<FloatArray>(prepared.size)

            // Group similar lengths together so short texts aren't padded up to the
            // longest text in the corpus. Original order is restored via the index.
            val order = prepared.indices.sortedByDescending { prepared[it].length }

            for (batch in order.chunked(spec.batchSize)) {
                // Each batch is hundreds of milliseconds of native work; checking
                // here is what makes "leave the screen" actually stop indexing.
                currentCoroutineContext().ensureActive()
                val encoded = tokenizer.encodeBatch(batch.map { prepared[it] }, spec.maxSequenceTokens)
                val vectors = mutex.withLock { runSession(encoded) }
                batch.forEachIndexed { i, originalIndex -> out[originalIndex] = vectors[i] }
            }

            out.map { requireNotNull(it) }
        }
    }

    private fun runSession(batch: TokenBatch): List<FloatArray> =
        OrtInputs.run(environment, session, batch, needsTokenTypeIds) { tensor ->
            val hidden = tensor.floatBuffer
            val hiddenSize = tensor.info.shape.last().toInt()
            check(hiddenSize == spec.dimensions) {
                "model outputs $hiddenSize dims but spec declares ${spec.dimensions} — wrong model file?"
            }
            pool(hidden, batch, hiddenSize)
        }

    /**
     * Masked mean pooling: average token vectors over real tokens only.
     *
     * Including [PAD] positions here is the single most common on-device
     * embedding bug — it works with batch size 1 and silently degrades as soon
     * as a batch contains texts of different lengths.
     */
    private fun pool(hidden: java.nio.FloatBuffer, batch: TokenBatch, hiddenSize: Int): List<FloatArray> {
        val out = ArrayList<FloatArray>(batch.batchSize)
        for (b in 0 until batch.batchSize) {
            val vector = FloatArray(hiddenSize)
            var counted = 0

            when (spec.pooling) {
                Pooling.CLS -> {
                    val base = b * batch.sequenceLength * hiddenSize
                    for (h in 0 until hiddenSize) vector[h] = hidden.get(base + h)
                    counted = 1
                }
                Pooling.MEAN -> {
                    for (t in 0 until batch.sequenceLength) {
                        if (batch.attentionMask[b * batch.sequenceLength + t] == 0L) continue
                        val base = (b * batch.sequenceLength + t) * hiddenSize
                        for (h in 0 until hiddenSize) vector[h] += hidden.get(base + h)
                        counted++
                    }
                }
            }

            if (counted > 0 && spec.pooling == Pooling.MEAN) {
                for (h in 0 until hiddenSize) vector[h] /= counted
            }
            out.add(Vectors.normalize(vector))
        }
        return out
    }

    override fun close() {
        session.close()
    }

    companion object {

        /**
         * @param tokenizerFile `vocab.txt`, `sentencepiece.bpe.model` or `unigram_vocab.tsv`,
         *   matching [EmbeddingModelSpec.tokenizer].
         * @param threads defaults to half the cores, capped at 4. Using every core
         *   starves the UI thread and on big.LITTLE hardware often runs slower.
         */
        fun create(
            context: Context,
            model: ModelSource,
            tokenizerFile: ModelSource,
            spec: EmbeddingModelSpec = EmbeddingModelSpec.ALL_MINILM_L6_V2,
            threads: Int = defaultThreads(),
            useNnapi: Boolean = false,
            dispatchers: RagDispatchers = RagDispatchers(),
        ): OnnxEmbedder {
            val tokenizer = Tokenizers.load(context, tokenizerFile, spec.tokenizer, spec.doLowerCase)
            val environment = OrtEnvironment.getEnvironment()
            val session = OrtSessions.create(context, environment, model, threads, useNnapi)
            return OnnxEmbedder(environment, session, tokenizer, spec, dispatchers)
        }

        fun defaultThreads(): Int = min(4, maxOf(1, Runtime.getRuntime().availableProcessors() / 2))
    }
}

object Tokenizers {
    fun load(context: Context, source: ModelSource, format: TokenizerFormat, doLowerCase: Boolean = true): Tokenizer =
        open(context, source).use { stream ->
            when (format) {
                TokenizerFormat.WORDPIECE_VOCAB ->
                    WordPieceTokenizer.fromVocabLines(stream.bufferedReader().readLines().asSequence(), doLowerCase)
                TokenizerFormat.SENTENCEPIECE_MODEL ->
                    UnigramTokenizer.fromSentencePieceModel(stream.readBytes(), UnigramTokenizer.IdMapping.FAIRSEQ)
                TokenizerFormat.UNIGRAM_TSV ->
                    UnigramTokenizer.fromTsv(stream.bufferedReader().readLines().asSequence())
            }
        }

    internal fun open(context: Context, source: ModelSource): InputStream = when (source) {
        is ModelSource.Asset -> context.assets.open(source.name)
        is ModelSource.LocalFile -> File(source.path).also {
            require(it.isFile) { "tokenizer file not found: ${source.path}" }
        }.inputStream()
    }
}

internal object OrtSessions {
    fun create(
        context: Context,
        environment: OrtEnvironment,
        model: ModelSource,
        threads: Int,
        useNnapi: Boolean,
    ): OrtSession {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (useNnapi) {
                // Off by default: NNAPI frequently falls back to CPU for int8
                // quantized graphs and on some vendors returns wrong results.
                // Turn it on only after the parity test passes on that device.
                addNnapi()
            }
        }
        return when (model) {
            is ModelSource.LocalFile -> {
                val file = File(model.path)
                require(file.isFile) { "model file not found: ${model.path}" }
                environment.createSession(file.absolutePath, options)
            }
            is ModelSource.Asset ->
                environment.createSession(context.assets.open(model.name).use { it.readBytes() }, options)
        }
    }
}

internal object OrtInputs {
    /** Creates the input tensors, runs the session, and always closes the tensors. */
    fun <T> run(
        environment: OrtEnvironment,
        session: OrtSession,
        batch: TokenBatch,
        includeTokenTypes: Boolean,
        read: (OnnxTensor) -> T,
    ): T {
        val shape = longArrayOf(batch.batchSize.toLong(), batch.sequenceLength.toLong())
        val inputs = HashMap<String, OnnxTensor>(3)
        try {
            inputs["input_ids"] = OnnxTensor.createTensor(environment, LongBuffer.wrap(batch.inputIds), shape)
            inputs["attention_mask"] = OnnxTensor.createTensor(environment, LongBuffer.wrap(batch.attentionMask), shape)
            if (includeTokenTypes) {
                inputs["token_type_ids"] = OnnxTensor.createTensor(environment, LongBuffer.wrap(batch.tokenTypeIds), shape)
            }
            return session.run(inputs).use { results -> read(results[0] as OnnxTensor) }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }
}
