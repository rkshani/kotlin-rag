package io.github.rkshani.rag.embed.onnx

import java.util.Base64
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The fixture model is a small Unigram model trained on mixed Urdu and English
 * with sentencepiece's `nmt_nfkc` rule, and `expected.tsv` holds the ids the
 * reference sentencepiece library produced for each input. Any divergence means
 * the Kotlin port disagrees with the reference — fix the port, never the fixture.
 */
class UnigramTokenizerTest {

    private fun resource(name: String) =
        requireNotNull(javaClass.getResourceAsStream("/unigram/$name")) { "missing test resource $name" }

    private val model = resource("urdu_english_test.model").use { it.readBytes() }
    private val spm = UnigramTokenizer.fromSentencePieceModel(model, UnigramTokenizer.IdMapping.SENTENCEPIECE)
    private val fairseq = UnigramTokenizer.fromSentencePieceModel(model, UnigramTokenizer.IdMapping.FAIRSEQ)

    @Test
    fun `matches the reference sentencepiece ids on every fixture`() {
        val lines = resource("expected.tsv").bufferedReader().readLines().filter { it.isNotEmpty() && !it.startsWith("#") }
        assertTrue(lines.size >= 20)
        val mismatches = lines.mapNotNull { line ->
            val (encoded, ids) = line.split('\t', limit = 2).let { it[0] to it.getOrElse(1) { "" } }
            val text = String(Base64.getDecoder().decode(encoded), Charsets.UTF_8)
            val expected = ids.split(' ').filter { it.isNotEmpty() }.map { it.toInt() }
            val actual = spm.encode(text).toList()
            if (actual == expected) null else "'$text'\n   expected $expected\n   actual   $actual  ${spm.tokenize(text)}"
        }
        assertTrue(mismatches.isEmpty(), "diverged from sentencepiece:\n" + mismatches.joinToString("\n"))
    }

    @Test
    fun `fairseq mapping matches the xlm-r id layout`() {
        assertContentEquals(
            intArrayOf(4, 5, 37, 4, 99, 65, 10, 9, 12, 4, 27, 8, 17, 37, 28),
            fairseq.encode("یہ اردو کا جملہ ہے۔"),
        )
    }

    @Test
    fun `pair encoding uses the double separator`() {
        val batch = fairseq.encodePairBatch("solar", listOf("بیٹری"), maxSequenceLength = 64)
        assertEquals(listOf(0L, 41, 49, 60, 2, 2, 50, 5, 81, 30, 5, 2), batch.inputIds.toList())
        assertTrue(batch.tokenTypeIds.all { it == 0L })
    }

    @Test
    fun `single text batch is wrapped in s and end tokens and padded`() {
        val batch = fairseq.encodeBatch(listOf("solar panels", "solar"), maxSequenceLength = 64)
        val rowLength = batch.sequenceLength
        assertEquals(0L, batch.inputIds[0])
        assertEquals(0L, batch.inputIds[rowLength])
        assertEquals(1L, batch.inputIds[2 * rowLength - 1], "second row should end in <pad>")
        assertEquals(0L, batch.attentionMask[2 * rowLength - 1])
    }

    @Test
    fun `special tokens typed in text are not treated as specials`() {
        val ids = fairseq.encode("</s> <s>")
        assertTrue(ids.none { it == 0 || it == 2 }, "literal </s> in text became a real separator: ${ids.toList()}")
    }

    @Test
    fun `truncation keeps both specials`() {
        val batch = fairseq.encodeBatch(listOf("solar ".repeat(200)), maxSequenceLength = 16)
        assertEquals(16, batch.sequenceLength)
        assertEquals(0L, batch.inputIds.first())
        assertEquals(2L, batch.inputIds.last())
    }

    @Test
    fun `pair truncation cuts the long passage and keeps the short query`() {
        val query = fairseq.encode("solar")
        val batch = fairseq.encodePairBatch("solar", listOf("بیٹری ".repeat(100)), maxSequenceLength = 20)
        assertEquals(20, batch.sequenceLength)
        assertContentEquals(query.map { it.toLong() }.toLongArray(), batch.inputIds.copyOfRange(1, 1 + query.size))
    }

    @Test
    fun `tsv vocab gives the same result as the model file`() {
        val lines = sequence {
            for (id in 0 until fairseq.vocabSize) yield(id)
        }
        // Round-trip through the exporter's format.
        val reader = SentencePieceModelReader.read(model)
        val tsv = buildList {
            add("<s>\t0\tCONTROL"); add("<pad>\t0\tCONTROL"); add("</s>\t0\tCONTROL"); add("<unk>\t0\tUNKNOWN")
            reader.pieces.drop(3).forEach { add("${it.text}\t${it.score}\t${it.type.name}") }
            add("<mask>\t0\tCONTROL")
        }
        val fromTsv = UnigramTokenizer.fromTsv(tsv.asSequence())
        val text = "mixed اردو and English 2024 ☃"
        assertContentEquals(fairseq.encode(text), fromTsv.encode(text))
        assertEquals(fairseq.vocabSize, lines.count())
    }

    @Test
    fun `blank input yields no ids`() {
        assertTrue(spm.encode("").isEmpty())
        assertTrue(spm.encode(" \t\n ").isEmpty())
    }
}
