package io.github.rkshani.rag.voice

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import java.io.File
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.min

/**
 * Offline speech-to-text with Whisper on ONNX Runtime.
 *
 * Expects the "all-in-one" Whisper model produced by Microsoft Olive's whisper
 * example (see `tools/prepare_whisper.py`): audio pre-processing, encoder,
 * decoder with beam search and token decoding in one graph, built from
 * onnxruntime-extensions custom ops. Kotlin only feeds audio in and reads text out.
 *
 * Handles both export variants by inspecting the model's inputs:
 * - `audio_pcm` (exported with `--no_audio_decoder`): float samples, 16 kHz mono.
 * - `audio_stream`: encoded audio bytes — we send a WAV file built in memory.
 * - `decoder_input_ids` (exported with `--multilingual`): the forced language
 *   prompt. Required for Urdu; see [WhisperPrompts].
 *
 * Model size guide for Urdu: tiny and base produce poor Urdu; small (int8,
 * ~250 MB) is the practical minimum.
 *
 * Not thread-safe; [WhisperVoiceInput] serialises calls.
 */
class WhisperOnnx private constructor(
    private val environment: OrtEnvironment,
    private val session: OrtSession,
    private val prompts: WhisperPrompts,
) : AutoCloseable {

    private val inputs: Set<String> = session.inputNames

    val isMultilingual: Boolean get() = "decoder_input_ids" in inputs

    init {
        require("audio_pcm" in inputs || "audio_stream" in inputs) {
            "not an all-in-one Whisper model: expected an audio_pcm or audio_stream input, found $inputs"
        }
    }

    /**
     * @param audio 16 kHz mono samples in -1..1. Whisper reads 30 s at most;
     *   anything longer is cut, never silently split.
     */
    fun transcribe(audio: FloatArray, language: VoiceLanguage, maxTokens: Int = 200, beams: Int = 1): String {
        if (audio.isEmpty()) return ""
        if (!isMultilingual && language != VoiceLanguage.ENGLISH) {
            error("this Whisper model is English-only; export it with --multilingual for ${language.name}")
        }
        val clipped = if (audio.size > MAX_SAMPLES) audio.copyOf(MAX_SAMPLES) else audio

        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            if ("audio_pcm" in inputs) {
                tensors["audio_pcm"] = OnnxTensor.createTensor(environment, FloatBuffer.wrap(clipped), longArrayOf(1, clipped.size.toLong()))
            } else {
                val wav = Pcm.wav(clipped)
                tensors["audio_stream"] = OnnxTensor.createTensor(environment, ByteBuffer.wrap(wav), longArrayOf(1, wav.size.toLong()), OnnxJavaType.UINT8)
            }
            tensors["max_length"] = int(min(maxTokens, 448))
            tensors["min_length"] = int(1)
            tensors["num_beams"] = int(beams)
            tensors["num_return_sequences"] = int(1)
            tensors["length_penalty"] = float(1f)
            tensors["repetition_penalty"] = float(1f)
            if ("attention_mask" in inputs) {
                tensors["attention_mask"] = OnnxTensor.createTensor(environment, IntBuffer.wrap(IntArray(80 * 3000)), longArrayOf(1, 80, 3000))
            }
            if (isMultilingual) {
                val prompt = prompts.forLanguage(language)
                tensors["decoder_input_ids"] = OnnxTensor.createTensor(environment, IntBuffer.wrap(prompt), longArrayOf(1, prompt.size.toLong()))
            }
            // Only pass what this export declares; extra inputs are an error in ORT.
            val feed = tensors.filterKeys { it in inputs }

            return session.run(feed).use { result ->
                @Suppress("UNCHECKED_CAST")
                val text = (result[0].value as Array<Array<String>>)[0][0]
                TranscriptText.clean(text)
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    private fun int(value: Int) = OnnxTensor.createTensor(environment, IntBuffer.wrap(intArrayOf(value)), longArrayOf(1))

    private fun float(value: Float) = OnnxTensor.createTensor(environment, FloatBuffer.wrap(floatArrayOf(value)), longArrayOf(1))

    override fun close() = session.close()

    companion object {
        private const val MAX_SAMPLES = 30 * AudioRecorder.SAMPLE_RATE

        /**
         * @param promptFile `whisper_prompts.txt` from prepare_whisper.py; when
         *   null, [WhisperPrompts.DEFAULT] is used (correct for tiny to medium,
         *   not for large-v3).
         */
        fun create(model: File, promptFile: File? = null, threads: Int = 4): WhisperOnnx {
            require(model.isFile) { "Whisper model not found: $model" }
            val prompts = promptFile?.takeIf { it.isFile }?.let { file -> WhisperPrompts.parse(file.readLines().asSequence()) }
                ?: WhisperPrompts.DEFAULT
            val environment = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                registerCustomOpLibrary(OrtxPackage.getLibraryPath())
                setIntraOpNumThreads(threads)
            }
            return WhisperOnnx(environment, environment.createSession(model.absolutePath, options), prompts)
        }
    }
}
