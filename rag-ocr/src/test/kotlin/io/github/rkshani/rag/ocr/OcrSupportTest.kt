package io.github.rkshani.rag.ocr

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class OcrTextTest {

    @Test
    fun `rejoins latin words hyphenated across lines`() {
        assertEquals("photovoltaic cells", OcrText.clean("photo-\nvoltaic cells"))
    }

    @Test
    fun `keeps urdu text and does not join across an urdu dash`() {
        val urdu = "شمسی توانائی -\nبیٹری"
        assertEquals("شمسی توانائی -\nبیٹری", OcrText.clean(urdu))
    }

    @Test
    fun `drops border and speck noise lines but keeps short real lines`() {
        val raw = "Chapter 3\n|||~~==__--||\nSolar panels\n.\n7\n_|_"
        assertEquals("Chapter 3\n\nSolar panels\n\n7", OcrText.clean(raw))
    }

    @Test
    fun `collapses spaces and keeps paragraph breaks`() {
        assertEquals("a b\n\nc", OcrText.clean("a    b\n\n\n\n\nc  "))
    }

    @Test
    fun `normalises line endings and form feeds`() {
        assertEquals("one\ntwo\nthree", OcrText.clean("one\r\ntwo\rthree"))
        assertEquals("page one\npage two", OcrText.clean("page one\u000Cpage two"))
    }

    @Test
    fun `content chars counts letters and digits in any script`() {
        assertEquals(8, OcrText.contentChars("اردو 2024 !!"))
        assertEquals(0, OcrText.contentChars(" \n|-_ "))
    }
}

class RenderSizeTest {

    @Test
    fun `small pages render at the requested dpi`() {
        // 3 x 2 inch label at 300 dpi.
        assertEquals(900 to 600, RenderSize.of(216, 144, dpi = 300, maxSide = 2480))
    }

    @Test
    fun `a4 is capped at the max side and keeps its aspect ratio`() {
        val (w, h) = RenderSize.of(595, 842, dpi = 300, maxSide = 2480)
        assertEquals(2480, h)
        assertEquals(1752, w) // 595 * 2480 / 842 = 1752.49
    }

    @Test
    fun `sample size keeps a 50 megapixel photo near the cap`() {
        assertEquals(2, RenderSize.sampleSize(8160, 6120, maxSide = 2480))
        assertEquals(1, RenderSize.sampleSize(2000, 1500, maxSide = 2480))
        assertEquals(4, RenderSize.sampleSize(12000, 9000, maxSide = 2480))
    }
}

class OcrLanguageTest {

    @Test
    fun `reads the per document language from metadata`() {
        assertEquals(OcrLanguage.URDU, OcrLanguage.fromMetadata(mapOf("ocrLanguage" to "urdu"), OcrLanguage.ENGLISH))
        assertEquals(OcrLanguage.MIXED, OcrLanguage.fromMetadata(mapOf("ocrLanguage" to "MIXED"), OcrLanguage.ENGLISH))
        assertEquals(OcrLanguage.ENGLISH, OcrLanguage.fromMetadata(mapOf("ocrLanguage" to "klingon"), OcrLanguage.ENGLISH))
        assertEquals(OcrLanguage.URDU, OcrLanguage.fromMetadata(emptyMap(), OcrLanguage.URDU))
    }

    @Test
    fun `mixed needs both traineddata files`() {
        assertEquals(listOf("urd.traineddata", "eng.traineddata"), TessDataNames.required(OcrLanguage.MIXED.tesseractCode))
    }

    @Test
    fun `only plain traineddata names are installed`() {
        assertTrue(TessDataNames.isValid("urd.traineddata"))
        assertTrue(TessDataNames.isValid("urd_nastaliq.traineddata"))
        assertFalse(TessDataNames.isValid("../urd.traineddata"))
        assertFalse(TessDataNames.isValid("urd.traineddata.part"))
        assertFalse(TessDataNames.isValid("URD.traineddata"))
    }
}
