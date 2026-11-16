package io.github.rkshani.rag.voice

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.sqrt

/** Pure helpers for voice input, free of Android types so they run in JVM unit tests. */

enum class VoiceLanguage(
    /** Whisper language code. */
    val whisperCode: String,
    /** BCP-47 tag for Android's speech recognizer. */
    val localeTag: String,
) {
    URDU("ur", "ur-PK"),
    ENGLISH("en", "en-US"),
}

/**
 * Decides when the speaker has finished, from the loudness of each audio frame.
 *
 * Tapping "stop" after every question is tedious, and Whisper reads a fixed
 * 30-second window, so recording also has to end on its own. The noise floor
 * is measured over the first frames instead of using a fixed threshold — a fan
 * or traffic would otherwise count as speech forever, and a quiet room would
 * never reach a loud fixed threshold.
 */
class Endpointer(
    private val frameMillis: Int = 30,
    private val calibrationMillis: Int = 300,
    private val minSpeechMillis: Int = 150,
    private val trailingSilenceMillis: Int = 1_200,
    private val noSpeechTimeoutMillis: Int = 7_000,
    private val maxMillis: Int = 29_000,
    /** Speech must be this many times louder than the measured noise floor. */
    private val speechToNoise: Float = 2.5f,
    /** Absolute floor so a digitally silent mic doesn't make every click "speech". */
    private val minSpeechRms: Float = 0.012f,
) {
    enum class Reason { END_OF_SPEECH, NO_SPEECH, MAX_LENGTH }

    private var elapsed = 0
    private var calibrationSum = 0.0
    private var calibrationFrames = 0
    private var noiseFloor = 0f
    private var loudRun = 0
    private var quietRun = 0

    var heardSpeech: Boolean = false
        private set

    /** Feed one frame's RMS (0..1). Returns non-null once recording should stop. */
    fun accept(rms: Float): Reason? {
        elapsed += frameMillis
        if (elapsed <= calibrationMillis) {
            calibrationSum += rms
            calibrationFrames++
            noiseFloor = (calibrationSum / calibrationFrames).toFloat()
            return null
        }
        val threshold = max(minSpeechRms, noiseFloor * speechToNoise)
        if (rms >= threshold) {
            loudRun += frameMillis
            quietRun = 0
            if (loudRun >= minSpeechMillis) heardSpeech = true
        } else {
            loudRun = 0
            quietRun += frameMillis
        }
        return when {
            elapsed >= maxMillis -> Reason.MAX_LENGTH
            heardSpeech && quietRun >= trailingSilenceMillis -> Reason.END_OF_SPEECH
            !heardSpeech && elapsed >= noSpeechTimeoutMillis -> Reason.NO_SPEECH
            else -> null
        }
    }
}

object Pcm {
    /** 16-bit PCM to floats in -1..1, the scale Whisper was trained on. */
    fun toFloat(samples: ShortArray, count: Int = samples.size): FloatArray =
        FloatArray(count) { samples[it] / 32768f }

    fun rms(samples: ShortArray, count: Int = samples.size): Float {
        if (count == 0) return 0f
        var sum = 0.0
        for (i in 0 until count) {
            val v = samples[i] / 32768.0
            sum += v * v
        }
        return sqrt(sum / count).toFloat()
    }

    /**
     * A complete mono 16-bit WAV file. Whisper models exported with an audio
     * decoder take encoded audio bytes rather than samples; WAV is the one
     * format such a decoder always reads.
     */
    fun wav(samples: FloatArray, sampleRate: Int = 16_000): ByteArray {
        val data = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        samples.forEach { data.putShort((it.coerceIn(-1f, 1f) * 32767f).toInt().toShort()) }
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(36 + samples.size * 2)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16) // PCM header size
            putShort(1) // PCM
            putShort(1) // mono
            putInt(sampleRate)
            putInt(sampleRate * 2) // byte rate
            putShort(2) // block align
            putShort(16) // bits per sample
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(samples.size * 2)
        }
        return ByteArrayOutputStream(44 + samples.size * 2).apply {
            write(header.array())
            write(data.array())
        }.toByteArray()
    }
}

/**
 * The forced decoder prompt per language: `<|startoftranscript|> <|lang|>
 * <|transcribe|> <|notimestamps|>`.
 *
 * Forcing the language matters for Urdu specifically: left to detect it,
 * Whisper often hears Urdu as Hindi and writes it in Devanagari, which then
 * matches nothing in an Urdu index.
 *
 * The ids differ between Whisper generations (large-v3 shifted the task
 * tokens), so `tools/prepare_whisper.py` writes them from the model's own
 * tokenizer. [DEFAULT] holds the ids for tiny / base / small / medium.
 */
class WhisperPrompts(private val byLanguage: Map<String, IntArray>) {

    fun forLanguage(language: VoiceLanguage): IntArray =
        byLanguage[language.whisperCode] ?: error("no Whisper prompt for '${language.whisperCode}'")

    companion object {
        val DEFAULT = WhisperPrompts(
            mapOf(
                "en" to intArrayOf(50258, 50259, 50359, 50363),
                "ur" to intArrayOf(50258, 50290, 50359, 50363),
            ),
        )

        /** Lines of `code id id id id`, as written by prepare_whisper.py. `#` starts a comment. */
        fun parse(lines: Sequence<String>): WhisperPrompts {
            val map = HashMap<String, IntArray>()
            lines.map { it.substringBefore('#').trim() }.filter { it.isNotEmpty() }.forEach { line ->
                val parts = line.split(Regex("""\s+"""))
                require(parts.size >= 2) { "bad prompt line: '$line'" }
                require(Regex("""[a-z]{2,3}""").matches(parts[0])) { "bad language code '${parts[0]}'" }
                map[parts[0]] = parts.drop(1).map { it.toInt() }.toIntArray()
            }
            require(map.isNotEmpty()) { "prompt file is empty" }
            return WhisperPrompts(map)
        }
    }
}

object TranscriptText {
    private val SPECIAL = Regex("""<\|[^|>]{1,40}\|>""")
    private val SPACES = Regex("""\s+""")

    /**
     * Strips special tokens that leak into decoded text and Whisper's stock
     * hallucinations on silence — the model was trained on subtitles, so near
     * silent input often decodes to a subtitle credit or "Thank you".
     */
    fun clean(raw: String): String {
        val text = SPACES.replace(SPECIAL.replace(raw, " "), " ").trim()
        val bare = text.trim('.', '!', ' ', '۔').lowercase()
        return if (bare.isEmpty() || bare in HALLUCINATIONS) "" else text
    }

    private val HALLUCINATIONS = setOf(
        "thank you", "thanks for watching", "thank you for watching", "you",
        "subtitles by the amara.org community", "شکریہ",
    )
}
