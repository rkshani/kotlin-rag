package io.github.rkshani.rag.embed.onnx

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * These expectations were produced by HuggingFace `tokenizers`
 * (BertWordPieceTokenizer, lowercase=true, strip_accents=true) over the same
 * synthetic vocab built below. If a change here starts failing, the Kotlin port
 * has drifted from the reference — fix the port, never the expectation.
 */
class WordPieceTokenizerTest {

    private val vocabTokens: List<String> = buildList {
        addAll(listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"))
        addAll(
            listOf(
                "the", "quick", "brown", "fox", "jump", "##s", "##ing", "##ed", "over", "lazy", "dog",
                "un", "##aff", "##able", "token", "##izer", "##ization", "play", "##ground",
                "e", "##m", "##b", "##ed", "##d", "##i", "##n", "##g", "hello", "world", "don", "'", "t",
                "cafe", "naive", "2024", "-", "!", ",", ".", "?", ":", ";", "(", ")", "\"", "#", "%",
                "北", "京", "ا", "##ر", "##د", "##و", "۔", "،",
            ),
        )
        ('a'..'z').forEach { add(it.toString()); add("##$it") }
        ('0'..'9').forEach { add(it.toString()); add("##$it") }
    }.distinct()

    private val tokenizer = WordPieceTokenizer.fromVocabLines(vocabTokens.asSequence())
    private val vocabIndex = vocabTokens.withIndex().associate { (i, t) -> t to i }

    @Test
    fun `splits known words into greedy longest-match subwords`() {
        assertEquals(
            listOf("token", "##ization", "i", "##s", "un", "##aff", "##able", "!"),
            tokenizer.tokenize("Tokenization is unaffable!"),
        )
    }

    @Test
    fun `lowercases and strips accents`() {
        assertEquals(listOf("cafe", "naive"), tokenizer.tokenize("CAFÉ naïve"))
    }

    @Test
    fun `splits on punctuation without dropping it`() {
        assertEquals(
            listOf("hello", ",", "world", "(", "2024", ")"),
            tokenizer.tokenize("hello,world (2024)"),
        )
    }

    @Test
    fun `apostrophes split into separate tokens`() {
        assertEquals(
            listOf("don", "'", "t", "s", "##t", "##o", "##p"),
            tokenizer.tokenize("don't  stop"),
        )
    }

    @Test
    fun `cjk characters are tokenized individually`() {
        assertEquals(listOf("北", "京", "hello"), tokenizer.tokenize("北京 hello"))
    }

    @Test
    fun `strips zero-width and non-breaking space`() {
        assertEquals(listOf("hello", "world", "!"), tokenizer.tokenize("Hello\u00a0World\u200b!"))
    }

    @Test
    fun `words longer than the char limit become UNK`() {
        assertEquals(listOf("[UNK]"), tokenizer.tokenize("a".repeat(150)))
    }

    @Test
    fun `unmatched characters make the whole word UNK`() {
        assertEquals(listOf("[UNK]"), tokenizer.tokenize("متن"))
    }

    @Test
    fun `blank input produces no tokens`() {
        assertTrue(tokenizer.tokenize("").isEmpty())
        assertTrue(tokenizer.tokenize("   \n\t ").isEmpty())
    }

    @Test
    fun `encode maps tokens to their vocab ids`() {
        val ids = tokenizer.encode("hello world")
        assertContentEquals(intArrayOf(vocabIndex.getValue("hello"), vocabIndex.getValue("world")), ids)
    }

    @Test
    fun `batch adds special tokens and pads to the longest row`() {
        val batch = tokenizer.encodeBatch(listOf("hello world", "hello"), maxSequenceLength = 128)

        assertEquals(2, batch.batchSize)
        assertEquals(4, batch.sequenceLength) // [CLS] hello world [SEP]

        val cls = vocabIndex.getValue("[CLS]").toLong()
        val sep = vocabIndex.getValue("[SEP]").toLong()
        val pad = vocabIndex.getValue("[PAD]").toLong()

        assertEquals(cls, batch.inputIds[0])
        assertEquals(sep, batch.inputIds[3])
        // Second row is one token shorter, so its last slot is padding.
        assertEquals(pad, batch.inputIds[7])
        assertContentEquals(longArrayOf(1, 1, 1, 1, 1, 1, 1, 0), batch.attentionMask)
        assertTrue(batch.tokenTypeIds.all { it == 0L })
    }

    @Test
    fun `batch truncates to the sequence limit leaving room for specials`() {
        val long = "hello ".repeat(200)
        val batch = tokenizer.encodeBatch(listOf(long), maxSequenceLength = 32)

        assertEquals(32, batch.sequenceLength)
        assertEquals(vocabIndex.getValue("[SEP]").toLong(), batch.inputIds[31])
        assertTrue(batch.attentionMask.all { it == 1L })
    }

    @Test
    fun `empty batch is handled`() {
        val batch = tokenizer.encodeBatch(emptyList(), maxSequenceLength = 128)
        assertEquals(0, batch.batchSize)
    }
}

class WordPiecePairTest {
    private val vocab = listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "solar", "battery", "night", "store", "energy")
    private val tokenizer = WordPieceTokenizer.fromVocabLines(vocab.asSequence())

    @Test
    fun `pairs use segment ids 0 then 1`() {
        val batch = tokenizer.encodePairBatch("solar", listOf("battery store energy"), maxSequenceLength = 32)
        kotlin.test.assertEquals(listOf(2L, 5, 3, 6, 8, 9, 3), batch.inputIds.toList())
        kotlin.test.assertEquals(listOf(0L, 0, 0, 1, 1, 1, 1), batch.tokenTypeIds.toList())
    }

    @Test
    fun `padding rows get mask and type zero`() {
        val batch = tokenizer.encodePairBatch("solar", listOf("battery store energy", "night"), maxSequenceLength = 32)
        kotlin.test.assertEquals(7, batch.sequenceLength)
        kotlin.test.assertEquals(listOf(1L, 1, 1, 1, 1, 0, 0), batch.attentionMask.toList().takeLast(7))
        kotlin.test.assertEquals(listOf(0L, 0, 0, 1, 1, 0, 0), batch.tokenTypeIds.toList().takeLast(7))
    }
}
