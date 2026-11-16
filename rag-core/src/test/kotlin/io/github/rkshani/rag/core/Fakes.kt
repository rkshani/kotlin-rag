package io.github.rkshani.rag.core

import io.github.rkshani.rag.core.text.SearchText
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.math.abs

/**
 * Deterministic bag-of-words embedder: each word hashes into one of [dimensions]
 * buckets. Texts sharing words get similar vectors, which is all the pipeline
 * tests need from "semantics".
 */
class HashingEmbedder(
    override val dimensions: Int = 64,
    override val maxSequenceTokens: Int = 256,
    private val failAfter: Int = Int.MAX_VALUE,
) : Embedder {
    var calls = 0
        private set
    var closed = false
        private set

    override suspend fun embed(texts: List<String>, kind: EmbeddingKind): List<FloatArray> {
        calls++
        if (calls > failAfter) error("embedder failure on call $calls")
        return texts.map { text ->
            val vector = FloatArray(dimensions)
            SearchText.normalize(text).split(Regex("""[^\p{L}\p{N}]+""")).filter { it.isNotEmpty() }.forEach {
                vector[abs(it.hashCode()) % dimensions] += 1f
            }
            io.github.rkshani.rag.core.math.Vectors.normalize(vector)
        }
    }

    override fun close() {
        closed = true
    }
}

class ScriptedGenerator(
    private val tokens: List<String>,
    private val available: Boolean = true,
    private val failAfterTokens: Int? = null,
    override val appliesChatTemplate: Boolean = false,
) : Generator {
    val prompts = mutableListOf<String>()
    var closed = false
        private set

    override suspend fun isAvailable() = available

    override fun generate(prompt: String, options: GenerationOptions): Flow<String> = flow {
        prompts.add(prompt)
        tokens.forEachIndexed { i, token ->
            if (failAfterTokens != null && i >= failAfterTokens) error("native engine crashed")
            emit(token)
        }
    }

    override fun close() {
        closed = true
    }
}
