package io.github.rkshani.rag.core.text

import java.text.Normalizer

/**
 * Folds text into the form used as a keyword-search key.
 *
 * SQLite's `unicode61` tokenizer only knows about Latin diacritics. Arabic-script
 * vowel marks (zer, zabar, pesh, tashdid ...) are combining characters it treats
 * as word separators, so a diacritised "کِتاب" is indexed as two fragments and a
 * plain query for "کتاب" never finds it. The same letter also has several code
 * points depending on the keyboard that typed it: Arabic yeh vs Urdu yeh, Arabic
 * kaf vs Urdu kaf, three different hehs.
 *
 * Both the indexed text and the query go through this function, so they only
 * have to agree with each other — the output is a search key, never shown to a
 * user. The original text is stored separately and untouched.
 */
object SearchText {

    fun normalize(text: String): String {
        if (text.isEmpty()) return text
        // NFKC first: it folds Arabic presentation forms (what PDF extraction
        // often produces) back to plain letters and composes alef + madda.
        val composed = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val out = StringBuilder(composed.length)
        var i = 0
        while (i < composed.length) {
            val cp = composed.codePointAt(i)
            i += Character.charCount(cp)
            when {
                isRemoved(cp) -> Unit
                else -> out.appendCodePoint(Character.toLowerCase(fold(cp)))
            }
        }
        return out.toString()
    }

    /** Marks that carry no meaning for matching: vowel signs, tatweel, joiners, bidi controls. */
    private fun isRemoved(cp: Int): Boolean =
        cp in 0x0610..0x061A || cp in 0x064B..0x065F || cp == 0x0670 ||
            cp in 0x06D6..0x06DC || cp in 0x06DF..0x06E4 || cp == 0x06E7 || cp == 0x06E8 ||
            cp in 0x06EA..0x06ED ||
            cp == 0x0640 || // tatweel (kashida)
            cp == 0x200C || cp == 0x200D || // zero-width non-joiner / joiner
            cp == 0x200E || cp == 0x200F || cp in 0x202A..0x202E || cp in 0x2066..0x2069 ||
            cp == 0xFEFF

    private fun fold(cp: Int): Int = when (cp) {
        0x064A, 0x0649 -> 0x06CC // Arabic yeh, alef maksura -> Urdu yeh
        0x0643 -> 0x06A9 // Arabic kaf -> keheh
        0x0647, 0x06D5, 0x06C0 -> 0x06C1 // Arabic heh, ae, heh with yeh above -> heh goal
        0x0629 -> 0x06C3 // teh marbuta -> teh marbuta goal
        0x06D3 -> 0x06D2 // yeh barree with hamza -> yeh barree
        0x0623, 0x0625 -> 0x0627 // alef with hamza above / below -> alef
        in 0x0660..0x0669 -> cp - 0x0660 + '0'.code // Arabic-Indic digits
        in 0x06F0..0x06F9 -> cp - 0x06F0 + '0'.code // Extended Arabic-Indic (Urdu) digits
        else -> cp
    }
}
