package io.github.rkshani.rag.demo

import android.app.Application
import android.content.Context
import android.util.Log
import io.github.rkshani.rag.core.KotlinRag
import io.github.rkshani.rag.embed.onnx.EmbeddingModelSpec
import io.github.rkshani.rag.embed.onnx.ModelSource
import io.github.rkshani.rag.embed.onnx.OnnxEmbedder
import io.github.rkshani.rag.llm.ModelSpec
import io.github.rkshani.rag.llm.VerifiedModelGenerator
import io.github.rkshani.rag.ocr.ImageOcrLoader
import io.github.rkshani.rag.ocr.MlKitOcr
import io.github.rkshani.rag.ocr.OcrPdf
import io.github.rkshani.rag.ocr.RoutingOcr
import io.github.rkshani.rag.ocr.TessData
import io.github.rkshani.rag.ocr.TesseractOcr
import io.github.rkshani.rag.store.room.RoomRagStore
import io.github.rkshani.rag.voice.SystemVoiceInput
import io.github.rkshani.rag.voice.VoiceInput
import io.github.rkshani.rag.voice.WhisperVoiceInput
import java.io.File

class DemoApp : Application() {
    val graph: AppGraph by lazy { AppGraph(this) }
}

/**
 * Wires the library together once per process. Manual DI keeps the demo
 * readable; a real app would put this in a Hilt module.
 */
class AppGraph(context: Context) {

    /** The optional LLM, configured through Gradle properties (see demo/build.gradle.kts). */
    val llm: ModelSpec? = runCatching {
        if (BuildConfig.LLM_URL.isBlank()) null
        else ModelSpec(
            url = BuildConfig.LLM_URL,
            fileName = BuildConfig.LLM_FILE,
            sha256 = BuildConfig.LLM_SHA256,
            sizeBytes = BuildConfig.LLM_SIZE.takeIf { it > 0 },
        )
    }.onFailure { Log.w(TAG, "LLM settings rejected: ${it.message}") }.getOrNull()

    /** Null when the embedding model is missing from assets; [setupError] says why. */
    val rag: KotlinRag?
    val setupError: String?

    /**
     * Voice input, best first. Whisper is used when its files are on the phone:
     * `adb push whisper_int8.onnx whisper_prompts.txt /sdcard/Android/data/<app id>/files/voice/`.
     */
    val voiceInputs: List<VoiceInput> = run {
        val voiceDir = File(context.getExternalFilesDir(null) ?: context.filesDir, "voice")
        listOf(
            WhisperVoiceInput(context, File(voiceDir, "whisper_int8.onnx"), File(voiceDir, "whisper_prompts.txt")),
            SystemVoiceInput(context),
        )
    }

    /** Last "model cannot run here" reason, shown on the model screen. */
    @Volatile var unavailableReason: String? = null
        private set

    init {
        var built: KotlinRag? = null
        var error: String? = null
        try {
            val assets = context.assets.list("").orEmpty().toSet()
            require("model_int8.onnx" in assets) { "assets/model_int8.onnx is missing" }

            val (spec, tokenizerFile) = when {
                "unigram_vocab.tsv" in assets -> EmbeddingModelSpec.MULTILINGUAL_E5_SMALL to "unigram_vocab.tsv"
                "vocab.txt" in assets -> EmbeddingModelSpec.ALL_MINILM_L6_V2 to "vocab.txt"
                else -> error("assets need vocab.txt or unigram_vocab.tsv next to the model")
            }
            // OCR: ML Kit for English; Tesseract for Urdu when assets/tessdata has urd.traineddata.
            val ocr = RoutingOcr(latin = MlKitOcr(), tesseract = TesseractOcr(TessData.install(context)))

            val embedder = OnnxEmbedder.create(
                context = context,
                model = ModelSource.Asset("model_int8.onnx"),
                tokenizerFile = ModelSource.Asset(tokenizerFile),
                spec = spec,
            )
            built = KotlinRag.builder()
                .embedder(embedder)
                // One database per embedding model: vectors from different models don't mix.
                .store(RoomRagStore.create(context, embedder.dimensions, name = "rag-${tokenizerFile.substringBefore('.')}.db"))
                .tokenEstimator(embedder.tokenEstimator())
                .addLoader(OcrPdf.loader(context, ocr)) // text-layer PDFs, OCR for scanned pages
                .addLoader(ImageOcrLoader(ocr)) // photos of notes and pages
                .generator(llm?.let { VerifiedModelGenerator(context, it, onUnavailable = { reason -> unavailableReason = reason }) })
                .onFailure { component, failure -> Log.w(TAG, "$component degraded", failure) }
                .build()
        } catch (failure: Throwable) {
            Log.e(TAG, "setup failed", failure)
            error = failure.message ?: failure.toString()
        }
        rag = built
        setupError = error
    }

    private companion object {
        const val TAG = "kotlin-rag-demo"
    }
}
