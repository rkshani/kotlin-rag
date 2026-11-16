package io.github.rkshani.rag.core.chunking

import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.SourceDocument
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.random.Random

class RecursiveChunkerTest {

    private val estimator = HeuristicTokenEstimator()
    private val chunker = RecursiveChunker(targetTokens = 400, overlapTokens = 50)

    private fun doc(text: String) = SourceDocument(id = "doc1", text = text)

    /** The single most important invariant: offsets must index the ORIGINAL text. */
    private fun assertOffsetsResolve(text: String, chunks: List<Chunk>) {
        chunks.forEach { c ->
            assertEquals(
                text.substring(c.charStart, c.charEnd), c.text,
                "chunk ${c.index} text does not match its own offsets",
            )
        }
    }

    /** Nothing meaningful may be dropped — whitespace is the only allowed loss. */
    private fun assertNoContentLost(text: String, chunks: List<Chunk>) {
        val covered = BooleanArray(text.length)
        chunks.forEach { c -> for (i in c.charStart until c.charEnd) covered[i] = true }
        val lost = text.indices.firstOrNull { !text[it].isWhitespace() && !covered[it] }
        assertTrue(
            lost == null,
            "content lost at offset $lost: '${text.substring(maxOf(0, (lost ?: 0) - 20), minOf(text.length, (lost ?: 0) + 20))}'",
        )
    }

    private fun assertWithinBudget(chunks: List<Chunk>, target: Int = 400) {
        chunks.forEach { c ->
            val tokens = estimator.estimate(c.text)
            // 15% headroom: the merge step packs whole atoms, it cannot cut mid-atom.
            assertTrue(tokens <= target * 1.15, "chunk ${c.index} is $tokens tokens, over budget")
        }
    }

    @Test
    fun `splits english prose and preserves every character`() {
        val paragraph = "The quick brown fox jumps over the lazy dog. ".repeat(12).trim()
        val text = List(8) { paragraph }.joinToString("\n\n")

        val chunks = chunker.chunk(doc(text))

        assertTrue(chunks.size > 1, "expected multiple chunks, got ${chunks.size}")
        assertOffsetsResolve(text, chunks)
        assertNoContentLost(text, chunks)
        assertWithinBudget(chunks)
    }

    @Test
    fun `handles urdu text without overflowing the token budget`() {
        val text = "یہ ایک آزمائشی جملہ ہے جو اردو زبان میں لکھا گیا ہے۔ ".repeat(40)

        val chunks = chunker.chunk(doc(text))

        assertTrue(chunks.isNotEmpty())
        assertOffsetsResolve(text, chunks)
        assertNoContentLost(text, chunks)
        assertWithinBudget(chunks)
    }

    @Test
    fun `urdu is estimated as more tokens than latin at equal length`() {
        val latin = "a".repeat(200)
        val urdu = "ا".repeat(200)
        assertTrue(
            estimator.estimate(urdu) > estimator.estimate(latin),
            "Arabic-script text must not be undercounted",
        )
    }

    @Test
    fun `falls back to hard splitting when no separator exists`() {
        val text = "word".repeat(5000)

        val chunks = chunker.chunk(doc(text))

        assertTrue(chunks.size > 1)
        assertOffsetsResolve(text, chunks)
        assertNoContentLost(text, chunks)
        assertWithinBudget(chunks)
    }

    @Test
    fun `consecutive chunks overlap`() {
        val text = "Sentence number one is here. ".repeat(200)

        val chunks = chunker.chunk(doc(text))

        assertTrue(chunks.size > 1)
        val overlapping = chunks.zipWithNext().count { (a, b) -> b.charStart < a.charEnd }
        assertEquals(chunks.size - 1, overlapping, "every adjacent pair should share context")
    }

    @Test
    fun `chunks advance strictly and are indexed in order`() {
        val text = "Alpha beta gamma delta. ".repeat(300)

        val chunks = chunker.chunk(doc(text))

        chunks.forEachIndexed { i, c -> assertEquals(i, c.index) }
        chunks.zipWithNext().forEach { (a, b) ->
            assertTrue(b.charStart >= a.charStart && b.charEnd > a.charEnd, "chunker did not advance")
        }
    }

    @Test
    fun `blank and whitespace-only documents produce nothing`() {
        assertTrue(chunker.chunk(doc("")).isEmpty())
        assertTrue(chunker.chunk(doc("   \n\n\t  ")).isEmpty())
    }

    @Test
    fun `short document becomes a single trimmed chunk`() {
        val chunks = chunker.chunk(doc("  short document  "))

        assertEquals(1, chunks.size)
        assertEquals("short document", chunks.first().text)
    }

    @Test
    fun `no tiny trailing fragment`() {
        val text = "Filler sentence goes here. ".repeat(150) + " end."

        val chunks = chunker.chunk(doc(text))

        val tail = estimator.estimate(chunks.last().text)
        assertTrue(chunks.size == 1 || tail >= 40, "trailing fragment of $tail tokens should have been folded")
    }

    @Test
    fun `survives random noise without losing content or hanging`() {
        val alphabet = ('a'..'z') + ('A'..'Z') + listOf(' ', '\n', '.', '،', '۔')
        val rng = Random(7)
        val text = buildString { repeat(20_000) { append(alphabet[rng.nextInt(alphabet.size)]) } }

        val chunks = chunker.chunk(doc(text))

        assertOffsetsResolve(text, chunks)
        assertNoContentLost(text, chunks)
        assertWithinBudget(chunks)
    }

    @Test
    fun `assigns page numbers from page offsets`() {
        val page1 = "First page content. ".repeat(30)
        val page2 = "Second page content. ".repeat(30)
        val document = SourceDocument(
            id = "paged",
            text = page1 + page2,
            pageOffsets = listOf(0, page1.length),
        )

        val chunks = RecursiveChunker(targetTokens = 60, overlapTokens = 10).chunk(document)

        assertEquals(1, chunks.first().page)
        val lastPage = chunks.last().page
        assertNotNull(lastPage)
        assertEquals(2, lastPage)
    }
}
