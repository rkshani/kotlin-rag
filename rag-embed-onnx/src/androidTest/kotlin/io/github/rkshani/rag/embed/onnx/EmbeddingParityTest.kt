package io.github.rkshani.rag.embed.onnx

import androidx.test.platform.app.InstrumentationRegistry
import io.github.rkshani.rag.core.EmbeddingKind
import io.github.rkshani.rag.core.math.Vectors
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Compares on-device output against fixtures produced by `tools/export_embedder.py`.
 *
 * This is the test that decides whether Phase 2 is actually done. A tokenizer or
 * pooling bug does not crash and does not look wrong in the UI — it just quietly
 * returns vectors that retrieve the wrong passages. Only a numeric comparison
 * against the Python reference catches it.
 *
 * Skips itself when the model has not been pushed to the device:
 *   adb push out/model_int8.onnx /sdcard/Android/data/<pkg>/files/model_int8.onnx
 *   adb push out/vocab.txt      /sdcard/Android/data/<pkg>/files/vocab.txt
 * For a multilingual (XLM-R) export, push `unigram_vocab.tsv` instead of vocab.txt;
 * the fixture's "tokenizer" field tells the test which one to load.
 */
class EmbeddingParityTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private val modelFile get() = File(context.getExternalFilesDir(null), "model_int8.onnx")
    private fun fixture(): JSONObject =
        InstrumentationRegistry.getInstrumentation().context.assets
            .open("parity.json").bufferedReader().use { JSONObject(it.readText()) }

    private fun format(fixture: JSONObject = fixture()) =
        TokenizerFormat.valueOf(fixture.optString("tokenizer", TokenizerFormat.WORDPIECE_VOCAB.name))

    private val vocabFile: File
        get() = File(
            context.getExternalFilesDir(null),
            if (format() == TokenizerFormat.UNIGRAM_TSV) "unigram_vocab.tsv" else "vocab.txt",
        )

    private fun spec(fixture: JSONObject) = EmbeddingModelSpec(
        dimensions = fixture.getInt("dimensions"),
        maxSequenceTokens = fixture.getInt("max_sequence_tokens"),
        doLowerCase = fixture.optBoolean("lowercase", true),
        tokenizer = format(fixture),
    )

    private fun embedder(@Suppress("UNUSED_PARAMETER") dimensions: Int, @Suppress("UNUSED_PARAMETER") maxTokens: Int) =
        OnnxEmbedder.create(
            context = context,
            model = ModelSource.LocalFile(modelFile.absolutePath),
            tokenizerFile = ModelSource.LocalFile(vocabFile.absolutePath),
            spec = spec(fixture()),
        )

    /** Compares full id rows, special tokens included, against Hugging Face's tokenizer. */
    @Test
    fun tokenizerMatchesPythonReference() {
        assumeTrue("tokenizer file not on device", vocabFile.exists())
        val fixture = fixture()
        val spec = spec(fixture)
        val tokenizer = Tokenizers.load(context, ModelSource.LocalFile(vocabFile.absolutePath), spec.tokenizer, spec.doLowerCase)

        val entries = fixture.getJSONArray("entries")
        for (i in 0 until entries.length()) {
            val entry = entries.getJSONObject(i)
            val text = entry.getString("text")
            val expected = entry.getJSONArray("ids").let { arr -> List(arr.length()) { arr.getLong(it) } }
            val actual = tokenizer.encodeBatch(listOf(text), spec.maxSequenceTokens).inputIds.toList()
            assertEquals(expected, actual, "tokenization diverged for: $text")
        }
    }

    @Test
    fun embeddingsMatchPythonReference() = runBlocking {
        assumeTrue("model not on device", modelFile.exists() && vocabFile.exists())
        val fixture = fixture()
        val dimensions = fixture.getInt("dimensions")
        val maxTokens = fixture.getInt("max_sequence_tokens")

        embedder(dimensions, maxTokens).use { embedder ->
            val entries = fixture.getJSONArray("entries")
            val texts = List(entries.length()) { entries.getJSONObject(it).getString("text") }

            // Deliberately embedded as ONE batch of mixed lengths — this is what
            // exposes mean-pooling over padding.
            val actual = embedder.embed(texts, EmbeddingKind.DOCUMENT)

            var worst = 1.0f
            actual.forEachIndexed { i, vector ->
                val expectedArray = entries.getJSONObject(i).getJSONArray("embedding")
                val expected = FloatArray(expectedArray.length()) { expectedArray.getDouble(it).toFloat() }

                assertEquals(dimensions, vector.size)
                val similarity = Vectors.cosine(vector, expected)
                worst = minOf(worst, similarity)
                assertTrue(
                    similarity > 0.999f,
                    "cosine ${"%.5f".format(similarity)} for '${texts[i]}' — tokenizer or pooling mismatch",
                )
            }
            android.util.Log.i("Parity", "worst cosine vs reference: $worst")
        }
    }

    @Test
    fun batchingDoesNotChangeResults() = runBlocking {
        assumeTrue("model not on device", modelFile.exists() && vocabFile.exists())
        val fixture = fixture()
        val dimensions = fixture.getInt("dimensions")
        val maxTokens = fixture.getInt("max_sequence_tokens")

        val texts = listOf(
            "short",
            "a considerably longer sentence that will pad the short one",
            "medium length input here",
        )

        embedder(dimensions, maxTokens).use { embedder ->
            val batched = embedder.embed(texts)
            val individually = texts.map { embedder.embed(listOf(it)).single() }

            batched.forEachIndexed { i, vector ->
                val similarity = Vectors.cosine(vector, individually[i])
                assertTrue(
                    similarity > 0.9999f,
                    "batch changed the vector for '${texts[i]}' (cosine $similarity) — padding is leaking into pooling",
                )
            }
        }
    }

    @Test
    fun vectorsAreNormalized() = runBlocking {
        assumeTrue("model not on device", modelFile.exists() && vocabFile.exists())
        val fixture = fixture()

        embedder(fixture.getInt("dimensions"), fixture.getInt("max_sequence_tokens")).use { embedder ->
            embedder.embed(listOf("normalization check", "second text")).forEach { vector ->
                assertTrue(kotlin.math.abs(Vectors.norm(vector) - 1f) < 1e-4f, "vector is not L2-normalised")
            }
        }
    }

    @Test
    fun semanticNeighboursRankAboveUnrelatedText() = runBlocking {
        assumeTrue("model not on device", modelFile.exists() && vocabFile.exists())
        val fixture = fixture()

        embedder(fixture.getInt("dimensions"), fixture.getInt("max_sequence_tokens")).use { embedder ->
            val query = embedder.embed("how do I run a model on a phone", EmbeddingKind.QUERY)
            val related = embedder.embed("on-device inference for mobile applications")
            val unrelated = embedder.embed("the recipe calls for two cups of flour")

            assertTrue(
                Vectors.cosine(query, related) > Vectors.cosine(query, unrelated),
                "embeddings carry no usable semantics — check pooling and the model file",
            )
        }
    }
}
