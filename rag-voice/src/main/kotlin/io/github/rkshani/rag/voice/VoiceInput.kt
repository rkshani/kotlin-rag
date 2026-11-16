package io.github.rkshani.rag.voice

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognitionSupport
import android.speech.RecognitionSupportCallback
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.resume

sealed interface VoiceEvent {
    data object Listening : VoiceEvent

    /** Loudness 0..1 for a level meter. */
    data class Level(val value: Float) : VoiceEvent

    data class Partial(val text: String) : VoiceEvent

    data object Transcribing : VoiceEvent

    /** The recognised question. Empty when nothing was said. */
    data class Final(val text: String) : VoiceEvent

    data class Failed(val message: String) : VoiceEvent
}

/**
 * Speak a question instead of typing it. Both implementations are offline:
 * nothing the user says leaves the phone, which is the whole promise of an
 * on-device RAG app.
 */
interface VoiceInput : AutoCloseable {
    val name: String

    suspend fun isAvailable(language: VoiceLanguage): Boolean

    /** Starts listening. Ends by itself when the speaker stops; [finish] ends it early. */
    fun listen(language: VoiceLanguage): Flow<VoiceEvent>

    fun finish()
}

/**
 * Whisper on device ([WhisperOnnx]). Best Urdu quality available offline, at
 * the cost of a 250 MB+ model and a second or more of transcription after each
 * question. The model is loaded on first use and kept.
 */
class WhisperVoiceInput(
    context: Context,
    private val modelFile: File,
    private val promptFile: File? = null,
) : VoiceInput {

    override val name: String = "Whisper (on device)"

    private val recorder = AudioRecorder(context.applicationContext)
    private val lock = Mutex()
    private var whisper: WhisperOnnx? = null

    /** Cheap check: file and permission. An English-only model asked for Urdu reports it from [listen]. */
    override suspend fun isAvailable(language: VoiceLanguage): Boolean = modelFile.isFile && recorder.hasPermission()

    override fun listen(language: VoiceLanguage): Flow<VoiceEvent> = flow {
        if (!recorder.hasPermission()) {
            emit(VoiceEvent.Failed("Microphone permission is needed"))
            return@flow
        }
        // Load before recording so the first question isn't followed by a long freeze.
        val model = loaded() ?: run {
            emit(VoiceEvent.Failed("Whisper model is missing"))
            return@flow
        }
        emit(VoiceEvent.Listening)
        var finished: RecordingEvent.Finished? = null
        recorder.record().collect { event ->
            when (event) {
                is RecordingEvent.Level -> emit(VoiceEvent.Level(event.value))
                is RecordingEvent.Finished -> finished = event
            }
        }
        val audio = finished
        if (audio == null || !audio.heardSpeech || audio.samples.isEmpty()) {
            emit(VoiceEvent.Final(""))
            return@flow
        }
        emit(VoiceEvent.Transcribing)
        try {
            val text = lock.withLock { withContext(Dispatchers.Default) { model.transcribe(audio.samples, language) } }
            emit(VoiceEvent.Final(text))
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            emit(VoiceEvent.Failed(failure.message ?: "transcription failed"))
        }
    }

    override fun finish() = recorder.finish()

    private suspend fun loaded(): WhisperOnnx? = lock.withLock {
        whisper ?: if (!modelFile.isFile) null else withContext(Dispatchers.IO) {
            WhisperOnnx.create(modelFile, promptFile)
        }.also { whisper = it }
    }

    override fun close() {
        whisper?.close()
        whisper = null
    }
}

/**
 * Android's own recognizer, strictly on-device (Android 12+). Light: no model
 * to ship. Urdu support depends on the phone — usually the user has to
 * download the Urdu offline pack once ([requestOfflineModel], Android 13+).
 *
 * This never falls back to the online recognizer: that would send the user's
 * voice to a server, which an offline document assistant must not do quietly.
 */
class SystemVoiceInput(private val context: Context) : VoiceInput {

    override val name: String = "Android speech (on device)"

    @Volatile private var active: SpeechRecognizer? = null

    override suspend fun isAvailable(language: VoiceLanguage): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return false
        if (!SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) return false
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return true // cannot ask per language before 13
        return support(language)?.installedOnDeviceLanguages?.any { it.matches(language) } == true
    }

    /** Asks the system to download the offline model for [language] (Android 13+). */
    fun requestOfflineModel(language: VoiceLanguage) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        try {
            recognizer.triggerModelDownload(intent(language))
        } finally {
            recognizer.destroy()
        }
    }

    // Every API-31+ call below sits behind the SDK_INT check at the top of the block.
    @SuppressLint("NewApi")
    override fun listen(language: VoiceLanguage): Flow<VoiceEvent> = callbackFlow {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S || !SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) {
            trySend(VoiceEvent.Failed("On-device speech recognition needs Android 12 or newer"))
            close()
            return@callbackFlow
        }
        val recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        active = recognizer
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                trySend(VoiceEvent.Listening)
            }

            override fun onRmsChanged(rmsdB: Float) {
                // Roughly -2..10 dB in practice.
                trySend(VoiceEvent.Level(((rmsdB + 2f) / 12f).coerceIn(0f, 1f)))
            }

            override fun onPartialResults(partialResults: Bundle?) {
                first(partialResults)?.let { trySend(VoiceEvent.Partial(it)) }
            }

            override fun onResults(results: Bundle?) {
                trySend(VoiceEvent.Final(first(results).orEmpty()))
                close()
            }

            override fun onError(error: Int) {
                val event = when (error) {
                    SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> VoiceEvent.Final("")
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> VoiceEvent.Failed("Microphone permission is needed")
                    SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED, SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE ->
                        VoiceEvent.Failed("${language.name.lowercase()} offline speech is not installed on this phone")
                    else -> VoiceEvent.Failed("speech recognition error $error")
                }
                trySend(event)
                close()
            }

            override fun onBeginningOfSpeech() = Unit
            override fun onBufferReceived(buffer: ByteArray?) = Unit
            override fun onEndOfSpeech() {
                trySend(VoiceEvent.Transcribing)
            }
            override fun onEvent(eventType: Int, params: Bundle?) = Unit
        })
        recognizer.startListening(intent(language))
        awaitClose {
            active = null
            recognizer.cancel()
            recognizer.destroy()
        }
    }.flowOn(Dispatchers.Main) // SpeechRecognizer must be used from the main thread

    override fun finish() {
        active?.stopListening()
    }

    override fun close() {
        active?.destroy()
        active = null
    }

    private fun intent(language: VoiceLanguage) = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, language.localeTag)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
    }

    private suspend fun support(language: VoiceLanguage): RecognitionSupport? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return null
        return withContext(Dispatchers.Main) {
            val recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
            try {
                suspendCancellableCoroutine { continuation ->
                    recognizer.checkRecognitionSupport(
                        intent(language),
                        context.mainExecutor,
                        object : RecognitionSupportCallback {
                            override fun onSupportResult(recognitionSupport: RecognitionSupport) {
                                if (continuation.isActive) continuation.resume(recognitionSupport)
                            }

                            override fun onError(error: Int) {
                                if (continuation.isActive) continuation.resume(null)
                            }
                        },
                    )
                }
            } finally {
                recognizer.destroy()
            }
        }
    }

    private fun String.matches(language: VoiceLanguage): Boolean =
        equals(language.localeTag, ignoreCase = true) || substringBefore('-').equals(language.whisperCode, ignoreCase = true)

    private fun first(bundle: Bundle?): String? =
        bundle?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim()?.takeIf { it.isNotEmpty() }
}

/** Picks Whisper when its model is installed, otherwise the system recognizer. */
object VoiceInputs {
    suspend fun best(candidates: List<VoiceInput>, language: VoiceLanguage): VoiceInput? =
        candidates.firstOrNull { runCatching { it.isAvailable(language) }.getOrDefault(false) }
}
