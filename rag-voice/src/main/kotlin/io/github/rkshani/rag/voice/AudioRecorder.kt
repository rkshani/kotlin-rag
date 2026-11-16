package io.github.rkshani.rag.voice

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.util.concurrent.atomic.AtomicBoolean

sealed interface RecordingEvent {
    /** Microphone loudness, 0..1, for a level meter. */
    data class Level(val value: Float) : RecordingEvent

    data class Finished(val samples: FloatArray, val reason: Endpointer.Reason?, val heardSpeech: Boolean) : RecordingEvent
}

/**
 * Records 16 kHz mono — what Whisper expects, so no resampling — and stops by
 * itself when the speaker finishes (see [Endpointer]), or when [finish] is called.
 *
 * Uses the VOICE_RECOGNITION source, which on most phones turns off the
 * automatic gain and noise processing tuned for calls; that processing distorts
 * speech in ways that hurt recognition.
 */
class AudioRecorder(private val context: Context) {

    private val finishRequested = AtomicBoolean(false)

    fun hasPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    /** Stop listening now and keep what was recorded. */
    fun finish() = finishRequested.set(true)

    @SuppressLint("MissingPermission") // checked by hasPermission() first
    fun record(endpointer: Endpointer = Endpointer()): Flow<RecordingEvent> = flow {
        check(hasPermission()) { "RECORD_AUDIO permission not granted" }
        finishRequested.set(false)

        val frameSamples = SAMPLE_RATE * FRAME_MILLIS / 1000
        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        check(minBuffer > 0) { "this device cannot record 16 kHz mono audio" }
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuffer, frameSamples * 2 * 4),
        )
        check(recorder.state == AudioRecord.STATE_INITIALIZED) { "microphone is unavailable (in use by another app?)" }

        val samples = ArrayList<FloatArray>()
        val frame = ShortArray(frameSamples)
        var reason: Endpointer.Reason? = null
        try {
            recorder.startRecording()
            while (reason == null && !finishRequested.get()) {
                // Leaving the screen cancels the collector; stop recording at once.
                currentCoroutineContext().ensureActive()
                val read = recorder.read(frame, 0, frame.size)
                if (read <= 0) continue
                samples.add(Pcm.toFloat(frame, read))
                val rms = Pcm.rms(frame, read)
                emit(RecordingEvent.Level((rms * 6f).coerceAtMost(1f)))
                reason = endpointer.accept(rms)
            }
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
        }

        val total = samples.sumOf { it.size }
        val audio = FloatArray(total)
        var offset = 0
        samples.forEach { chunk ->
            chunk.copyInto(audio, offset)
            offset += chunk.size
        }
        emit(RecordingEvent.Finished(audio, reason, endpointer.heardSpeech))
    }.flowOn(Dispatchers.IO)

    companion object {
        const val SAMPLE_RATE = 16_000
        private const val FRAME_MILLIS = 30
    }
}
