package io.github.rkshani.rag.embed.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import io.github.rkshani.rag.core.RagDispatchers
import io.github.rkshani.rag.core.retrieval.Reranker
import io.github.rkshani.rag.core.retrieval.RetrievedChunk
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlin.math.exp

data class CrossEncoderSpec(
    val maxSequenceTokens: Int = 512,
    val batchSize: Int = 4,
    val tokenizer: TokenizerFormat = TokenizerFormat.WORDPIECE_VOCAB,
    val doLowerCase: Boolean = true,
) {
    companion object {
        /** cross-encoder/ms-marco-MiniLM-L-6-v2 — English, ~23 MB int8, fast. */
        val MS_MARCO_MINILM_L6 = CrossEncoderSpec(maxSequenceTokens = 512)

        /**
         * cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 — multilingual including
         * Urdu, XLM-R tokenizer, ~120 MB int8. Roughly 4x the cost per pair of
         * the English model; keep `rerankCandidates` small.
         */
        val MMARCO_MULTILINGUAL_MINILM = CrossEncoderSpec(
            maxSequenceTokens = 512,
            tokenizer = TokenizerFormat.UNIGRAM_TSV,
        )
    }
}

/**
 * Re-ranks retrieved chunks with a cross-encoder: the query and each passage are
 * read together in one pass, which judges "does this answer the question" far
 * better than comparing two separately made embeddings.
 *
 * Scores are mapped through a sigmoid so they read as 0..1 relevance.
 */
class OnnxCrossEncoderReranker private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    private val tokenizer: Tokenizer,
    private val spec: CrossEncoderSpec,
    private val dispatchers: RagDispatchers,
) : Reranker, AutoCloseable {

    private val mutex = Mutex()
    private val needsTokenTypeIds = "token_type_ids" in session.inputNames

    override suspend fun rerank(query: String, candidates: List<RetrievedChunk>): List<RetrievedChunk> {
        if (candidates.size < 2) return candidates
        return withContext(dispatchers.default) {
            val scores = FloatArray(candidates.size)
            candidates.indices.chunked(spec.batchSize).forEach { batch ->
                currentCoroutineContext().ensureActive()
                val encoded = tokenizer.encodePairBatch(query, batch.map { candidates[it].chunk.text }, spec.maxSequenceTokens)
                val logits = mutex.withLock { score(encoded) }
                batch.forEachIndexed { i, index -> scores[index] = logits[i] }
            }
            candidates.indices
                .sortedByDescending { scores[it] }
                .map { candidates[it].copy(score = scores[it], rerankScore = scores[it]) }
        }
    }

    private fun score(batch: TokenBatch): FloatArray =
        OrtInputs.run(environment, session, batch, needsTokenTypeIds) { tensor ->
            val buffer = tensor.floatBuffer
            val columns = tensor.info.shape.last().toInt().coerceAtLeast(1)
            FloatArray(batch.batchSize) { row ->
                // One logit is a relevance score; two are (irrelevant, relevant).
                val logit = if (columns == 1) {
                    buffer.get(row)
                } else {
                    buffer.get(row * columns + columns - 1) - buffer.get(row * columns)
                }
                (1.0 / (1.0 + exp(-logit.toDouble()))).toFloat()
            }
        }

    override fun close() {
        session.close()
    }

    companion object {
        fun create(
            context: Context,
            model: ModelSource,
            tokenizerFile: ModelSource,
            spec: CrossEncoderSpec = CrossEncoderSpec.MS_MARCO_MINILM_L6,
            threads: Int = OnnxEmbedder.defaultThreads(),
            dispatchers: RagDispatchers = RagDispatchers(),
        ): OnnxCrossEncoderReranker {
            val tokenizer = Tokenizers.load(context, tokenizerFile, spec.tokenizer, spec.doLowerCase)
            val environment = OrtEnvironment.getEnvironment()
            val session = OrtSessions.create(context, environment, model, threads, useNnapi = false)
            return OnnxCrossEncoderReranker(environment, session, tokenizer, spec, dispatchers)
        }
    }
}
