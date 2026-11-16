package io.github.rkshani.rag.ocr

import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Which script a document is in. OCR engines are per-script: ML Kit reads Latin
 * only, Tesseract needs the right `.traineddata`. Guessing wrong produces
 * confident garbage, so the language is chosen per document.
 */
enum class OcrLanguage(val tesseractCode: String) {
    ENGLISH("eng"),
    URDU("urd"),

    /** Urdu and English on the same page — common in Pakistani notes and papers. */
    MIXED("urd+eng"),
    ;

    companion object {
        /** Put this key in `DocumentInput.metadata` to choose the language per document. */
        const val METADATA_KEY: String = "ocrLanguage"

        fun fromMetadata(metadata: Map<String, String>, default: OcrLanguage): OcrLanguage =
            metadata[METADATA_KEY]?.let { value -> entries.firstOrNull { it.name.equals(value, ignoreCase = true) } } ?: default
    }
}

class OcrUnavailableException(message: String) : java.io.IOException(message)

/** Pure helpers, kept free of Android types so they run in JVM unit tests. */
object OcrText {

    private val SPACES = Regex("""[ \t ]{2,}""")
    private val BLANK_LINES = Regex("""\n{3,}""")
    private val LATIN_HYPHEN_BREAK = Regex("""(\p{IsLatin})-\n(\p{IsLatin})""")

    /**
     * Cleans raw OCR output for indexing.
     *
     * - Words hyphenated across a line end are rejoined ("exam-\nple"). Latin
     *   only: Urdu is not hyphenated that way, and a dash there is real.
     * - Lines that are mostly not letters or digits are dropped: specks, page
     *   borders and table rules come out of OCR as `|~=_.-` noise, and noise
     *   lines in the index match random keyword queries.
     * - Runs of spaces collapse; paragraph breaks (blank lines) are kept,
     *   because the chunker splits on them first.
     */
    fun clean(raw: String, minLetterRatio: Double = 0.4): String {
        val unified = raw.replace("\r\n", "\n").replace('\r', '\n').replace('\u000C', '\n')
        val joined = LATIN_HYPHEN_BREAK.replace(unified, "$1$2")
        val lines = joined.split('\n').map { line ->
            val trimmed = SPACES.replace(line, " ").trim()
            if (trimmed.isEmpty() || isMostlyText(trimmed, minLetterRatio)) trimmed else ""
        }
        return BLANK_LINES.replace(lines.joinToString("\n"), "\n\n").trim()
    }

    private fun isMostlyText(line: String, minLetterRatio: Double): Boolean {
        val visible = line.count { !it.isWhitespace() }
        if (visible == 0) return true
        val letters = line.count { it.isLetterOrDigit() }
        // Short lines like "7" or "A." are real (page numbers, list markers).
        if (visible <= 2) return letters > 0
        return letters.toDouble() / visible >= minLetterRatio
    }

    /** Letters and digits in [text]: how much a page actually yielded. */
    fun contentChars(text: String): Int = text.count { it.isLetterOrDigit() }
}

/** Bitmap size for rendering a PDF page for OCR. */
object RenderSize {

    /**
     * PDF pages are measured in points (1/72 inch). OCR wants roughly 300 dpi,
     * but an A4 page at 300 dpi is 2480 x 3508 x 4 bytes = 35 MB of ARGB —
     * enough to kill a low-end phone. So the longest side is capped at
     * [maxSide], trading a little accuracy for staying alive.
     */
    fun of(widthPoints: Int, heightPoints: Int, dpi: Int = 300, maxSide: Int = 2480): Pair<Int, Int> {
        require(widthPoints > 0 && heightPoints > 0) { "page has no size" }
        require(dpi > 0 && maxSide > 0) { "dpi and maxSide must be positive" }
        var scale = dpi / 72.0
        val longest = max(widthPoints, heightPoints) * scale
        if (longest > maxSide) scale *= maxSide / longest
        val width = (widthPoints * scale).roundToInt().coerceAtLeast(1)
        val height = (heightPoints * scale).roundToInt().coerceAtLeast(1)
        return width to height
    }

    /** Power-of-two sample size so a decoded photo's longest side stays near [maxSide]. */
    fun sampleSize(width: Int, height: Int, maxSide: Int = 2480): Int {
        var sample = 1
        while (max(width, height) / (sample * 2) >= maxSide) sample *= 2
        return sample
    }
}

/** Validates traineddata names before they become file paths. */
object TessDataNames {
    private val VALID = Regex("""[a-z]{3}(_[a-z]+)?\.traineddata""")

    fun isValid(fileName: String): Boolean = VALID.matches(fileName)

    /** "urd+eng" -> ["urd.traineddata", "eng.traineddata"] */
    fun required(languageCode: String): List<String> = languageCode.split('+').filter { it.isNotBlank() }.map { "$it.traineddata" }
}
