package io.github.rkshani.rag.voice

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EndpointerTest {

    private fun Endpointer.feed(rms: Float, millis: Int): Endpointer.Reason? {
        var result: Endpointer.Reason? = null
        repeat(millis / 30) { if (result == null) result = accept(rms) }
        return result
    }

    @Test
    fun `stops after speech followed by silence`() {
        val endpointer = Endpointer()
        assertNull(endpointer.feed(0.005f, 300)) // calibration: quiet room
        assertNull(endpointer.feed(0.1f, 1500)) // speaking
        assertTrue(endpointer.heardSpeech)
        assertEquals(Endpointer.Reason.END_OF_SPEECH, endpointer.feed(0.005f, 1500))
    }

    @Test
    fun `short pauses between words do not end the recording`() {
        val endpointer = Endpointer()
        endpointer.feed(0.005f, 300)
        endpointer.feed(0.1f, 600)
        assertNull(endpointer.feed(0.005f, 600), "a 0.6 s pause ended the question")
        assertNull(endpointer.feed(0.1f, 600))
    }

    @Test
    fun `gives up when nobody speaks`() {
        val endpointer = Endpointer()
        assertEquals(Endpointer.Reason.NO_SPEECH, endpointer.feed(0.005f, 8000))
        assertFalse(endpointer.heardSpeech)
    }

    @Test
    fun `a noisy room raises the threshold instead of counting noise as speech`() {
        val endpointer = Endpointer()
        endpointer.feed(0.05f, 300) // fan noise during calibration
        assertNull(endpointer.feed(0.06f, 3000)) // same noise continues
        assertFalse(endpointer.heardSpeech)
        endpointer.feed(0.3f, 600) // actual speech, well above the noise
        assertTrue(endpointer.heardSpeech)
    }

    @Test
    fun `a click is not speech`() {
        val endpointer = Endpointer()
        endpointer.feed(0.005f, 300)
        endpointer.accept(0.5f) // one 30 ms frame
        endpointer.feed(0.005f, 300)
        assertFalse(endpointer.heardSpeech)
    }

    @Test
    fun `never records past the whisper window`() {
        val endpointer = Endpointer()
        endpointer.feed(0.005f, 300)
        assertEquals(Endpointer.Reason.MAX_LENGTH, endpointer.feed(0.2f, 31_000))
    }
}

class PcmTest {

    @Test
    fun `converts 16 bit samples to the -1 to 1 range`() {
        assertContentEquals(floatArrayOf(0f, 0.5f, -1f), Pcm.toFloat(shortArrayOf(0, 16384, -32768)))
    }

    @Test
    fun `rms of silence is zero and of full scale square wave is one`() {
        assertEquals(0f, Pcm.rms(ShortArray(100)))
        assertEquals(1f, Pcm.rms(ShortArray(100) { if (it % 2 == 0) -32768 else -32768 }), 1e-4f)
    }

    @Test
    fun `writes a valid wav header`() {
        val wav = Pcm.wav(FloatArray(1600) { 0.25f })
        val buffer = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", String(wav, 0, 4))
        assertEquals("WAVE", String(wav, 8, 4))
        assertEquals(36 + 3200, buffer.getInt(4))
        assertEquals(1, buffer.getShort(20).toInt()) // PCM
        assertEquals(1, buffer.getShort(22).toInt()) // mono
        assertEquals(16_000, buffer.getInt(24))
        assertEquals(16, buffer.getShort(34).toInt())
        assertEquals(3200, buffer.getInt(40))
        assertEquals(44 + 3200, wav.size)
        assertEquals(8191, buffer.getShort(44).toInt())
    }

    @Test
    fun `clips out of range samples instead of wrapping`() {
        val buffer = ByteBuffer.wrap(Pcm.wav(floatArrayOf(2f, -2f))).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(32767, buffer.getShort(44).toInt())
        assertEquals(-32767, buffer.getShort(46).toInt())
    }
}

class WhisperPromptsTest {

    @Test
    fun `default urdu prompt forces the urdu token`() {
        // ur is index 31 in Whisper's language list: 50259 + 31.
        assertContentEquals(intArrayOf(50258, 50290, 50359, 50363), WhisperPrompts.DEFAULT.forLanguage(VoiceLanguage.URDU))
        assertContentEquals(intArrayOf(50258, 50259, 50359, 50363), WhisperPrompts.DEFAULT.forLanguage(VoiceLanguage.ENGLISH))
    }

    @Test
    fun `parses the file written by the export script`() {
        val prompts = WhisperPrompts.parse(
            sequenceOf("# openai/whisper-large-v3", "en 50258 50259 50360 50364", "", "ur 50258 50290 50360 50364  # urdu"),
        )
        assertContentEquals(intArrayOf(50258, 50290, 50360, 50364), prompts.forLanguage(VoiceLanguage.URDU))
    }

    @Test
    fun `rejects malformed files`() {
        assertFailsWith<IllegalArgumentException> { WhisperPrompts.parse(sequenceOf("")) }
        assertFailsWith<IllegalArgumentException> { WhisperPrompts.parse(sequenceOf("urdu")) }
        assertFailsWith<NumberFormatException> { WhisperPrompts.parse(sequenceOf("ur 50258 abc")) }
        assertFailsWith<IllegalArgumentException> { WhisperPrompts.parse(sequenceOf("../x 1 2")) }
    }
}

class TranscriptTextTest {

    @Test
    fun `strips leaked special tokens and extra spaces`() {
        assertEquals("بیٹری کتنی دیر چلتی ہے؟", TranscriptText.clean("<|ur|><|transcribe|> بیٹری  کتنی دیر چلتی ہے؟ <|endoftext|>"))
    }

    @Test
    fun `drops typical silence hallucinations`() {
        assertEquals("", TranscriptText.clean(" Thank you."))
        assertEquals("", TranscriptText.clean("Thanks for watching!"))
        assertEquals("", TranscriptText.clean("شکریہ۔"))
        assertEquals("", TranscriptText.clean("  "))
    }

    @Test
    fun `keeps real questions that contain those words`() {
        assertEquals("Thank you, but how long do batteries last?", TranscriptText.clean("Thank you, but how long do batteries last?"))
    }
}
