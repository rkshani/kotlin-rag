package io.github.rkshani.rag.core.answer

import io.github.rkshani.rag.core.text.SearchText

enum class InjectionPolicy {
    /** Passages that read like instructions to the model are kept out of the prompt. */
    WITHHOLD,

    /** Everything goes in; only the structural defences apply. */
    ALLOW,
}

/**
 * Defences against text inside indexed documents steering the model.
 *
 * Everything retrieved is pasted into the prompt, so whoever wrote a document
 * gets to write part of the prompt. There is no complete fix for that — the
 * model reads instructions and data through the same channel. What this does is
 * close the cheap, reliable attacks and flag the obvious ones:
 *
 * 1. **Control tokens are removed.** A document containing
 *    `<end_of_turn><start_of_turn>user` would otherwise close the real user
 *    turn and open a forged one. This is the most dangerous attack on a small
 *    model and the only one that can be shut completely.
 * 2. **Passage delimiters cannot be forged.** Passage text is fenced in
 *    `<passage>` tags; a document cannot end its own fence early.
 * 3. **The system text tells the model passages are data.** Soft — small
 *    models obey it inconsistently — but cheap.
 * 4. **Instruction-shaped passages are detected** in English, Urdu and Roman
 *    Urdu, and by default withheld. Heuristic: it catches copy-pasted attacks,
 *    not a determined one.
 */
object PromptSafety {

    private val CONTROL_TOKENS = Regex(
        """<\|[A-Za-z_][A-Za-z0-9_]{0,30}\|>|<\s*/?\s*(?:start_of_turn|end_of_turn|bos|eos|s|im_start|im_end)\s*>|\[/?INST]|<</?SYS>>""",
        RegexOption.IGNORE_CASE,
    )

    private val PASSAGE_TAG = Regex("""<\s*/?\s*passage\b[^>]*>""", RegexOption.IGNORE_CASE)

    /** Matched against [SearchText]-normalised text, so case and Urdu letter variants are already folded. */
    private val INJECTION_PATTERNS = listOf(
        Regex("""\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system|these)\b.{0,30}\b(instructions?|prompts?|rules|directions|guidelines|context)\b"""),
        Regex("""\b(new|updated|real|actual)\s+(system\s+)?(instructions?|prompt)\s*:"""),
        Regex("""\byou\s+are\s+now\s+(a|an|the|in)\b"""),
        Regex("""\b(reveal|print|show|repeat|output)\b.{0,30}\b(system\s+prompt|hidden\s+instructions?|your\s+instructions?)\b"""),
        Regex("""\b(do\s+not|don'?t)\s+(follow|obey)\b.{0,30}\b(instructions?|rules)\b"""),
        Regex("""\b(pichl[ie]|pehl[ie]|purani)\s+(hidayat|instructions?)\b.{0,30}\b(ignore|nazar\s*andaz|bhool)"""),
        // Urdu, written after SearchText folding (Arabic yeh/kaf/heh already mapped to Urdu forms).
        Regex("""(پچھلی|پہلی|تمام|سابقہ)\s+(ہدایات|ہدایت)\s*(کو)?\s*(نظر\s*انداز|بھول)"""),
        Regex("""(ہدایات|ہدایت)\s*(کو)?\s*نظر\s*انداز\s*(کرو|کریں|کر\s*دو)"""),
    )

    fun stripControlTokens(text: String): String = CONTROL_TOKENS.replace(text, " ")

    /** Removes anything that could open or close a passage fence. */
    fun neutralizeDelimiters(text: String): String = PASSAGE_TAG.replace(text, " ")

    /** Everything that goes into a prompt from outside: document text, titles, past answers. */
    fun sanitize(text: String): String = neutralizeDelimiters(stripControlTokens(text))

    fun looksLikeInjection(text: String): Boolean {
        if (text.isBlank()) return false
        if (CONTROL_TOKENS.containsMatchIn(text)) return true
        val folded = SearchText.normalize(text).replace(Regex("""\s+"""), " ")
        return INJECTION_PATTERNS.any { it.containsMatchIn(folded) }
    }
}
