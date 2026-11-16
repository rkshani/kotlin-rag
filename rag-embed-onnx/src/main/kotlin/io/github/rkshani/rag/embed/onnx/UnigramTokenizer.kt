package io.github.rkshani.rag.embed.onnx

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer

/**
 * SentencePiece Unigram tokenizer in pure Kotlin — the tokenizer behind XLM-R
 * and every model built on it, including multilingual-e5 and the multilingual
 * MiniLM variants. This is what makes Urdu embeddings possible: WordPiece
 * vocabularies from English BERT have almost no Arabic-script pieces.
 *
 * Verified against the reference `sentencepiece` library on English, Urdu, mixed
 * script, digits, unknown characters, and whitespace edge cases (see the test
 * fixture). The pieces of the algorithm that must match exactly:
 *
 * - **Normalisation** is `nmt_nfkc`: NFKC plus a short list of control
 *   characters dropped and invisible separators turned into spaces. The list
 *   was measured by running the reference normaliser over every code point.
 * - **Whitespace**: trimmed, runs collapsed, a `▁` prefix added, spaces become `▁`.
 * - **Segmentation** is Viterbi over piece scores — the most probable split, not
 *   greedy longest-match.
 * - **Unknown characters** cost the lowest piece score minus 10, and a run of
 *   them collapses into one `<unk>`.
 *
 * Special pieces (`<s>`, `</s>`, `<pad>` ...) are never matched inside text, so a
 * document containing the literal string `</s>` cannot inject a separator.
 */
class UnigramTokenizer(
    pieces: List<Piece>,
    private val unkId: Int,
    private val bosId: Int,
    private val eosId: Int,
    private val padId: Int,
    private val addDummyPrefix: Boolean = true,
    private val removeExtraWhitespaces: Boolean = true,
    /** XLM-R separates a pair with two end tokens: `<s> A </s></s> B </s>`. */
    private val doubleSeparatorInPairs: Boolean = true,
) : Tokenizer {

    enum class PieceType { NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE }

    data class Piece(val text: String, val score: Float, val type: PieceType = PieceType.NORMAL)

    private class Node {
        val children = HashMap<Char, Node>(4)
        var id = -1
    }

    private val root = Node()
    private val pieceTexts: Array<String> = Array(pieces.size) { pieces[it].text }
    private val scores = FloatArray(pieces.size) { pieces[it].score }
    private val unkScore: Float

    override val vocabSize: Int = pieces.size

    init {
        require(unkId in pieces.indices) { "unkId $unkId is outside the vocab" }
        var minScore = Float.MAX_VALUE
        pieces.forEachIndexed { id, piece ->
            // Only normal and user-defined pieces can be produced from text.
            if (piece.type != PieceType.NORMAL && piece.type != PieceType.USER_DEFINED) return@forEachIndexed
            if (piece.text.isEmpty()) return@forEachIndexed
            var node = root
            for (ch in piece.text) node = node.children.getOrPut(ch) { Node() }
            node.id = id
            if (piece.type == PieceType.NORMAL) minScore = minOf(minScore, piece.score)
        }
        require(minScore != Float.MAX_VALUE) { "vocab has no normal pieces" }
        unkScore = minScore - UNK_PENALTY
    }

    override fun tokenize(text: String): List<String> = encode(text).map { pieceTexts[it] }

    override fun encode(text: String): IntArray {
        val normalized = normalize(text)
        if (normalized.isEmpty()) return IntArray(0)
        return viterbi(normalized)
    }

    override fun encodeBatch(texts: List<String>, maxSequenceLength: Int): TokenBatch {
        require(maxSequenceLength >= 2) { "maxSequenceLength must leave room for <s> and </s>" }
        if (texts.isEmpty()) return TokenBatch.EMPTY
        val bodyLimit = maxSequenceLength - 2
        val rows = texts.map { text ->
            val ids = encode(text)
            val body = if (ids.size > bodyLimit) ids.copyOf(bodyLimit) else ids
            IntArray(body.size + 2).also {
                it[0] = bosId
                body.copyInto(it, 1)
                it[body.size + 1] = eosId
            }
        }
        return padBatch(rows, types = null, padId = padId)
    }

    override fun encodePairBatch(first: String, seconds: List<String>, maxSequenceLength: Int): TokenBatch {
        val specials = if (doubleSeparatorInPairs) 4 else 3
        require(maxSequenceLength > specials) { "maxSequenceLength must leave room for $specials special tokens" }
        if (seconds.isEmpty()) return TokenBatch.EMPTY
        val firstIds = encode(first)
        val rows = seconds.map { second ->
            val (a, b) = truncatePair(firstIds, encode(second), maxSequenceLength - specials)
            buildList(a.size + b.size + specials) {
                add(bosId)
                a.forEach(::add)
                add(eosId)
                if (doubleSeparatorInPairs) add(eosId)
                b.forEach(::add)
                add(eosId)
            }.toIntArray()
        }
        // XLM-R has no segment embeddings; token types stay zero.
        return padBatch(rows, types = null, padId = padId)
    }

    internal fun normalize(text: String): String {
        val composed = Normalizer.normalize(text, Normalizer.Form.NFKC)
        val mapped = StringBuilder(composed.length)
        var i = 0
        while (i < composed.length) {
            val cp = composed.codePointAt(i)
            i += Character.charCount(cp)
            when {
                cp in REMOVED -> Unit
                cp in AS_SPACE || Character.getType(cp) == Character.SPACE_SEPARATOR.toInt() -> mapped.append(' ')
                cp == 0xFF5E -> mapped.append('～') // left alone by nmt_nfkc, unlike NFKC
                else -> mapped.appendCodePoint(cp)
            }
        }

        var body = mapped.toString()
        if (removeExtraWhitespaces) body = body.trim(' ').replace(MULTI_SPACE, " ")
        if (body.isEmpty()) return ""
        val escaped = body.replace(' ', SPACE_SYMBOL)
        return if (addDummyPrefix) SPACE_SYMBOL + escaped else escaped
    }

    private fun viterbi(text: String): IntArray {
        val n = text.length
        val best = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }
        val fromStart = IntArray(n + 1) { -1 }
        val fromId = IntArray(n + 1) { -1 }
        best[0] = 0f

        var begin = 0
        while (begin < n) {
            val base = best[begin]
            val charLen = Character.charCount(text.codePointAt(begin))
            if (base != Float.NEGATIVE_INFINITY) {
                var node: Node? = root
                var pos = begin
                var hasSingleCharPiece = false
                while (pos < n) {
                    node = node!!.children[text[pos]] ?: break
                    pos++
                    if (node.id >= 0) {
                        if (pos - begin == charLen) hasSingleCharPiece = true
                        val score = base + scores[node.id]
                        if (score > best[pos]) {
                            best[pos] = score
                            fromStart[pos] = begin
                            fromId[pos] = node.id
                        }
                    }
                }
                if (!hasSingleCharPiece) {
                    val end = begin + charLen
                    val score = base + unkScore
                    if (score > best[end]) {
                        best[end] = score
                        fromStart[end] = begin
                        fromId[end] = unkId
                    }
                }
            }
            begin += charLen
        }

        val reversed = ArrayList<Int>()
        var pos = n
        while (pos > 0) {
            val id = fromId[pos]
            // Collapse a run of unknown characters into one <unk>, as sentencepiece does.
            if (!(id == unkId && reversed.lastOrNull() == unkId)) reversed.add(id)
            pos = fromStart[pos]
        }
        val out = IntArray(reversed.size)
        for (i in out.indices) out[i] = reversed[reversed.size - 1 - i]
        return out
    }

    enum class IdMapping {
        /** Ids exactly as in the .model file. */
        SENTENCEPIECE,

        /**
         * fairseq / Hugging Face XLM-R ids: `<s>`=0, `<pad>`=1, `</s>`=2,
         * `<unk>`=3, every other piece shifted up by one, `<mask>` last.
         * This is what multilingual-e5 and other XLM-R exports expect.
         */
        FAIRSEQ,
    }

    companion object {
        const val SPACE_SYMBOL: Char = '▁'
        private const val UNK_PENALTY = 10f
        private val MULTI_SPACE = Regex(" {2,}")

        /** Measured against sentencepiece's `nmt_nfkc` rule over every code point. */
        private val REMOVED: Set<Int> = buildSet {
            addAll(0x01..0x08)
            add(0x0B)
            addAll(0x0E..0x1F)
            add(0x7F)
            add(0x8F)
            add(0x9F)
        }
        private val AS_SPACE: Set<Int> = setOf(
            0x09, 0x0A, 0x0C, 0x0D, 0x061C, 0x1680, 0x200B, 0x200C, 0x200E, 0x200F, 0x2028, 0x2029, 0xFEFF, 0xFFFD,
        )

        /**
         * Reads a sentencepiece `.model` file — the `sentencepiece.bpe.model`
         * shipped with XLM-R models (a Unigram model despite the name).
         */
        fun fromSentencePieceModel(input: InputStream, idMapping: IdMapping = IdMapping.FAIRSEQ): UnigramTokenizer =
            fromSentencePieceModel(input.use { it.readBytes() }, idMapping)

        fun fromSentencePieceModel(bytes: ByteArray, idMapping: IdMapping = IdMapping.FAIRSEQ): UnigramTokenizer {
            val model = SentencePieceModelReader.read(bytes)
            val pieces = model.pieces
            return when (idMapping) {
                IdMapping.SENTENCEPIECE -> UnigramTokenizer(
                    pieces = pieces,
                    unkId = pieces.indexOfFirst { it.type == PieceType.UNKNOWN }.also { require(it >= 0) { "model has no unknown piece" } },
                    bosId = pieces.indexOfFirst { it.text == "<s>" }.coerceAtLeast(0),
                    eosId = pieces.indexOfFirst { it.text == "</s>" }.coerceAtLeast(0),
                    padId = pieces.indexOfFirst { it.text == "<pad>" }.let { if (it >= 0) it else 0 },
                    addDummyPrefix = model.addDummyPrefix,
                    removeExtraWhitespaces = model.removeExtraWhitespaces,
                )
                IdMapping.FAIRSEQ -> {
                    // fairseq reserves 0..3; sentencepiece's own <unk>, <s>, </s> sit at 0..2.
                    val shifted = ArrayList<Piece>(pieces.size + 2)
                    shifted.add(Piece("<s>", 0f, PieceType.CONTROL))
                    shifted.add(Piece("<pad>", 0f, PieceType.CONTROL))
                    shifted.add(Piece("</s>", 0f, PieceType.CONTROL))
                    shifted.add(Piece("<unk>", 0f, PieceType.UNKNOWN))
                    pieces.forEachIndexed { i, piece ->
                        if (i < 3) return@forEachIndexed
                        shifted.add(piece)
                    }
                    shifted.add(Piece("<mask>", 0f, PieceType.CONTROL))
                    UnigramTokenizer(
                        pieces = shifted,
                        unkId = 3,
                        bosId = 0,
                        eosId = 2,
                        padId = 1,
                        addDummyPrefix = model.addDummyPrefix,
                        removeExtraWhitespaces = model.removeExtraWhitespaces,
                    )
                }
            }
        }

        /**
         * Reads the `unigram_vocab.tsv` written by `tools/export_embedder.py`:
         * one `piece<TAB>score<TAB>type` line per id, ids already as the model
         * expects them. Smaller than the .model file and needs no proto parsing.
         */
        fun fromTsv(lines: Sequence<String>): UnigramTokenizer {
            val pieces = ArrayList<Piece>(250_000)
            lines.forEach { line ->
                if (line.isEmpty()) return@forEach
                val parts = line.split('\t')
                require(parts.size >= 2) { "bad vocab line: '$line'" }
                val type = parts.getOrNull(2)?.let { PieceType.valueOf(it) } ?: PieceType.NORMAL
                pieces.add(Piece(unescape(parts[0]), parts[1].toFloat(), type))
            }
            fun id(text: String) = pieces.indexOfFirst { it.text == text }
            return UnigramTokenizer(
                pieces = pieces,
                unkId = pieces.indexOfFirst { it.type == PieceType.UNKNOWN }.also { require(it >= 0) { "vocab has no UNKNOWN piece" } },
                bosId = id("<s>").coerceAtLeast(0),
                eosId = id("</s>").coerceAtLeast(0),
                padId = id("<pad>").coerceAtLeast(0),
            )
        }

        private fun unescape(text: String): String =
            if ('\\' !in text) text else text.replace("\\t", "\t").replace("\\n", "\n").replace("\\\\", "\\")
    }
}

/**
 * Just enough protobuf to read a sentencepiece ModelProto: the piece list and
 * the two normaliser flags that change tokenization. Avoids a protobuf runtime.
 */
internal object SentencePieceModelReader {

    class Model(
        val pieces: List<UnigramTokenizer.Piece>,
        val addDummyPrefix: Boolean,
        val removeExtraWhitespaces: Boolean,
    )

    fun read(bytes: ByteArray): Model {
        val pieces = ArrayList<UnigramTokenizer.Piece>()
        var addDummyPrefix = true
        var removeExtraWhitespaces = true
        val reader = Reader(bytes, 0, bytes.size)
        while (reader.hasMore()) {
            val tag = reader.varint().toInt()
            val field = tag ushr 3
            val wire = tag and 7
            when {
                field == 1 && wire == 2 -> pieces.add(readPiece(reader.sub()))
                field == 3 && wire == 2 -> {
                    val spec = reader.sub()
                    while (spec.hasMore()) {
                        val t = spec.varint().toInt()
                        when {
                            t ushr 3 == 3 && t and 7 == 0 -> addDummyPrefix = spec.varint() != 0L
                            t ushr 3 == 4 && t and 7 == 0 -> removeExtraWhitespaces = spec.varint() != 0L
                            else -> spec.skip(t and 7)
                        }
                    }
                }
                else -> reader.skip(wire)
            }
        }
        require(pieces.isNotEmpty()) { "not a sentencepiece model: no pieces found" }
        return Model(pieces, addDummyPrefix, removeExtraWhitespaces)
    }

    private fun readPiece(reader: Reader): UnigramTokenizer.Piece {
        var text = ""
        var score = 0f
        var type = UnigramTokenizer.PieceType.NORMAL
        while (reader.hasMore()) {
            val tag = reader.varint().toInt()
            when {
                tag == (1 shl 3 or 2) -> text = String(reader.bytes(), Charsets.UTF_8)
                tag == (2 shl 3 or 5) -> score = reader.float()
                tag == (3 shl 3 or 0) -> type = when (reader.varint().toInt()) {
                    2 -> UnigramTokenizer.PieceType.UNKNOWN
                    3 -> UnigramTokenizer.PieceType.CONTROL
                    4 -> UnigramTokenizer.PieceType.USER_DEFINED
                    5 -> UnigramTokenizer.PieceType.UNUSED
                    6 -> UnigramTokenizer.PieceType.BYTE
                    else -> UnigramTokenizer.PieceType.NORMAL
                }
                else -> reader.skip(tag and 7)
            }
        }
        return UnigramTokenizer.Piece(text, score, type)
    }

    private class Reader(private val data: ByteArray, private var pos: Int, private val end: Int) {
        fun hasMore() = pos < end

        fun varint(): Long {
            var result = 0L
            var shift = 0
            while (true) {
                require(pos < end) { "truncated varint" }
                val b = data[pos++].toInt()
                result = result or ((b and 0x7F).toLong() shl shift)
                if (b and 0x80 == 0) return result
                shift += 7
                require(shift < 64) { "malformed varint" }
            }
        }

        fun bytes(): ByteArray {
            val length = varint().toInt()
            require(length >= 0 && pos + length <= end) { "truncated field" }
            return data.copyOfRange(pos, pos + length).also { pos += length }
        }

        fun sub(): Reader {
            val length = varint().toInt()
            require(length >= 0 && pos + length <= end) { "truncated message" }
            return Reader(data, pos, pos + length).also { pos += length }
        }

        fun float(): Float {
            require(pos + 4 <= end) { "truncated float" }
            return ByteBuffer.wrap(data, pos, 4).order(ByteOrder.LITTLE_ENDIAN).float.also { pos += 4 }
        }

        fun skip(wire: Int) {
            when (wire) {
                0 -> varint()
                1 -> pos += 8
                2 -> {
                    // Read the length first: `pos += varint()` would capture pos
                    // before varint() advances it, and drop the length bytes.
                    val length = varint().toInt()
                    require(length >= 0) { "negative field length" }
                    pos += length
                }
                5 -> pos += 4
                else -> error("unsupported wire type $wire")
            }
            require(pos <= end) { "truncated field" }
        }
    }
}
