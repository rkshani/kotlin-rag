package io.github.rkshani.rag.embed.onnx

import java.text.Normalizer

/**
 * BERT WordPiece tokenizer, implemented in pure Kotlin so the library carries no
 * native tokenizer dependency (~4 MB of .so per ABI avoided).
 *
 * This is a line-for-line port of a reference implementation validated against
 * HuggingFace `tokenizers` BertWordPieceTokenizer — accents, NBSP/zero-width
 * removal, CJK spacing, punctuation splitting, the >100-char fallback and the
 * greedy longest-match-first subword loop all agree exactly.
 *
 * If you ever swap in a model whose tokenizer is SentencePiece Unigram
 * (XLM-R based: multilingual MiniLM, multilingual-E5, LaBSE), this class does
 * NOT apply — implement [Tokenizer] with a Unigram Viterbi decoder instead.
 */
class WordPieceTokenizer(
    private val vocab: Map<String, Int>,
    private val doLowerCase: Boolean = true,
    private val unkToken: String = "[UNK]",
    private val clsToken: String = "[CLS]",
    private val sepToken: String = "[SEP]",
    private val padToken: String = "[PAD]",
    private val maxCharsPerWord: Int = 100,
) : Tokenizer {

    private val unkId = requireId(unkToken)
    private val clsId = requireId(clsToken)
    private val sepId = requireId(sepToken)
    private val padId = requireId(padToken)

    override val vocabSize: Int get() = vocab.size

    private fun requireId(token: String): Int =
        vocab[token] ?: error("vocab is missing required special token '$token'")

    override fun tokenize(text: String): List<String> {
        val out = ArrayList<String>()
        for (word in basicTokenize(text)) wordPiece(word, out)
        return out
    }

    override fun encode(text: String): IntArray {
        val pieces = tokenize(text)
        val ids = IntArray(pieces.size)
        for (i in pieces.indices) ids[i] = vocab[pieces[i]] ?: unkId
        return ids
    }

    override fun encodeBatch(texts: List<String>, maxSequenceLength: Int): TokenBatch {
        require(maxSequenceLength >= 2) { "maxSequenceLength must leave room for [CLS] and [SEP]" }
        if (texts.isEmpty()) return TokenBatch.EMPTY

        val bodyLimit = maxSequenceLength - 2
        val rows = texts.map { text ->
            val ids = encode(text)
            val body = if (ids.size > bodyLimit) ids.copyOf(bodyLimit) else ids
            IntArray(body.size + 2).also {
                it[0] = clsId
                body.copyInto(it, destinationOffset = 1)
                it[body.size + 1] = sepId
            }
        }
        return padBatch(rows, types = null, padId = padId)
    }

    /** `[CLS] first [SEP] second [SEP]`, token types 0 for the first segment and 1 for the second. */
    override fun encodePairBatch(first: String, seconds: List<String>, maxSequenceLength: Int): TokenBatch {
        require(maxSequenceLength >= 3) { "maxSequenceLength must leave room for three special tokens" }
        if (seconds.isEmpty()) return TokenBatch.EMPTY

        val firstIds = encode(first)
        val rows = ArrayList<IntArray>(seconds.size)
        val types = ArrayList<IntArray>(seconds.size)
        seconds.forEach { second ->
            val (a, b) = truncatePair(firstIds, encode(second), maxSequenceLength - 3)
            val row = IntArray(a.size + b.size + 3)
            val type = IntArray(row.size)
            row[0] = clsId
            a.copyInto(row, 1)
            row[a.size + 1] = sepId
            b.copyInto(row, a.size + 2)
            row[row.size - 1] = sepId
            for (i in a.size + 2 until row.size) type[i] = 1
            rows.add(row)
            types.add(type)
        }
        return padBatch(rows, types, padId)
    }

    // ---- Basic (pre-)tokenization ------------------------------------------------

    private fun basicTokenize(text: String): List<String> {
        val prepared = addCjkSpacing(cleanText(text))
        val out = ArrayList<String>()
        for (raw in prepared.split(' ')) {
            if (raw.isEmpty()) continue
            val normalized = if (doLowerCase) stripAccents(raw.lowercase()) else raw
            splitOnPunctuation(normalized, out)
        }
        return out
    }

    /** Drops control/format/surrogate/unassigned code points, folds all spacing to ' '. */
    private fun cleanText(text: String): String {
        val sb = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            val width = Character.charCount(cp)
            when {
                cp == 0 || cp == 0xFFFD || isControl(cp) -> Unit
                isWhitespace(cp) -> sb.append(' ')
                else -> sb.appendCodePoint(cp)
            }
            i += width
        }
        return sb.toString()
    }

    /** CJK ideographs are tokenized one character at a time, as BERT does. */
    private fun addCjkSpacing(text: String): String {
        val sb = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            if (isCjk(cp)) {
                sb.append(' ').appendCodePoint(cp).append(' ')
            } else {
                sb.appendCodePoint(cp)
            }
            i += Character.charCount(cp)
        }
        return sb.toString()
    }

    private fun stripAccents(text: String): String {
        val decomposed = Normalizer.normalize(text, Normalizer.Form.NFD)
        val sb = StringBuilder(decomposed.length)
        var i = 0
        while (i < decomposed.length) {
            val cp = decomposed.codePointAt(i)
            if (Character.getType(cp) != Character.NON_SPACING_MARK.toInt()) sb.appendCodePoint(cp)
            i += Character.charCount(cp)
        }
        return sb.toString()
    }

    private fun splitOnPunctuation(token: String, out: MutableList<String>) {
        val current = StringBuilder()
        var i = 0
        while (i < token.length) {
            val cp = token.codePointAt(i)
            if (isPunctuation(cp)) {
                if (current.isNotEmpty()) {
                    out.add(current.toString())
                    current.setLength(0)
                }
                out.add(String(Character.toChars(cp)))
            } else {
                current.appendCodePoint(cp)
            }
            i += Character.charCount(cp)
        }
        if (current.isNotEmpty()) out.add(current.toString())
    }

    // ---- WordPiece ---------------------------------------------------------------

    /** Greedy longest-match-first. Any unmatched remainder makes the WHOLE word [UNK]. */
    private fun wordPiece(word: String, out: MutableList<String>) {
        if (word.length > maxCharsPerWord) {
            out.add(unkToken)
            return
        }
        val pieces = ArrayList<String>(4)
        var start = 0
        while (start < word.length) {
            var end = word.length
            var match: String? = null
            while (start < end) {
                val candidate = if (start > 0) "##" + word.substring(start, end) else word.substring(start, end)
                if (vocab.containsKey(candidate)) {
                    match = candidate
                    break
                }
                end--
            }
            if (match == null) {
                out.add(unkToken)
                return
            }
            pieces.add(match)
            start = end
        }
        out.addAll(pieces)
    }

    // ---- Character classes -------------------------------------------------------

    private fun isControl(cp: Int): Boolean {
        if (cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return false
        return when (Character.getType(cp).toByte()) {
            Character.CONTROL, Character.FORMAT, Character.PRIVATE_USE,
            Character.SURROGATE, Character.UNASSIGNED,
            -> true
            else -> false
        }
    }

    private fun isWhitespace(cp: Int): Boolean {
        if (cp == ' '.code || cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return true
        return Character.getType(cp).toByte() == Character.SPACE_SEPARATOR
    }

    private fun isPunctuation(cp: Int): Boolean {
        // ASCII symbols that Unicode classifies as S* but BERT treats as punctuation.
        if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
        return when (Character.getType(cp).toByte()) {
            Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION,
            Character.START_PUNCTUATION, Character.END_PUNCTUATION,
            Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
            Character.OTHER_PUNCTUATION,
            -> true
            else -> false
        }
    }

    private fun isCjk(cp: Int): Boolean =
        cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF || cp in 0x20000..0x2A6DF ||
            cp in 0x2A700..0x2B73F || cp in 0x2B740..0x2B81F || cp in 0x2B820..0x2CEAF ||
            cp in 0xF900..0xFAFF || cp in 0x2F800..0x2FA1F

    companion object {
        /** Loads a standard `vocab.txt` — one token per line, id == line number. */
        fun fromVocabLines(lines: Sequence<String>, doLowerCase: Boolean = true): WordPieceTokenizer {
            val map = HashMap<String, Int>(32_000)
            var i = 0
            lines.forEach { line ->
                val token = line.trimEnd('\n', '\r')
                if (token.isNotEmpty()) map[token] = i
                i++
            }
            require(map.isNotEmpty()) { "vocab is empty" }
            return WordPieceTokenizer(map, doLowerCase)
        }
    }
}
