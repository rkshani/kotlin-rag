package io.github.rkshani.rag.core.retrieval

import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.ScoredChunk
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Expected scores come from an independent implementation of the same formula. */
class ReciprocalRankFusionTest {

    private fun chunk(id: String, docId: String = "doc1") = Chunk(
        id = id, docId = docId, index = 0, text = id, charStart = 0, charEnd = id.length,
    )

    private fun list(source: RetrievalSource, weight: Double = 1.0, vararg ids: String) = RankedList(
        source = source,
        results = ids.mapIndexed { i, id -> ScoredChunk(chunk(id), 1f - i * 0.1f) },
        weight = weight,
    )

    @Test
    fun `agreement across sources outranks a single strong hit`() {
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                list(RetrievalSource.VECTOR, 1.0, "a", "b", "c"),
                list(RetrievalSource.KEYWORD, 1.0, "b", "d", "a"),
            ),
        )

        assertEquals(listOf("b", "a", "d", "c"), fused.map { it.chunk.id })
        assertEquals(0.032522f, fused[0].score, 1e-6f)
        assertEquals(0.032266f, fused[1].score, 1e-6f)
    }

    @Test
    fun `second place in both lists beats first place in one`() {
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                list(RetrievalSource.VECTOR, 1.0, "x", "y"),
                list(RetrievalSource.KEYWORD, 1.0, "z", "y"),
            ),
        )

        assertEquals("y", fused.first().chunk.id)
        assertEquals(0.032258f, fused.first().score, 1e-6f)
    }

    @Test
    fun `weights shift the balance between sources`() {
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                list(RetrievalSource.VECTOR, 1.0, "a", "b"),
                list(RetrievalSource.KEYWORD, 2.0, "b", "a"),
            ),
        )

        assertEquals("b", fused.first().chunk.id, "the double-weighted source should decide the tie")
        assertEquals(0.048916f, fused[0].score, 1e-6f)
    }

    @Test
    fun `a deep vector hit is promoted when keywords rank it first`() {
        // The literal-token case: vectors bury the exact match at 15, BM25 finds
        // it immediately. Fusion pulls it into the top 3 — though note it does NOT
        // take first place, because a rank-15 contribution stays small at k=60.
        val vectorIds = (1..14).map { "v$it" } + "target"
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                RankedList(
                    RetrievalSource.VECTOR,
                    vectorIds.mapIndexed { i, id -> ScoredChunk(chunk(id), 0.9f - i * 0.01f) },
                ),
                list(RetrievalSource.KEYWORD, 1.0, "target", "v3"),
            ),
        )

        val position = fused.indexOfFirst { it.chunk.id == "target" }
        assertTrue(position in 0..2, "expected target in the top 3, was at $position")
        assertEquals(0.029727f, fused[position].score, 1e-6f)
    }

    @Test
    fun `provenance records the rank and score from every source`() {
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                list(RetrievalSource.VECTOR, 1.0, "a", "b"),
                list(RetrievalSource.KEYWORD, 1.0, "b"),
            ),
        )

        val b = fused.single { it.chunk.id == "b" }
        assertEquals(mapOf(RetrievalSource.VECTOR to 2, RetrievalSource.KEYWORD to 1), b.ranks)
        assertEquals(0.9f, b.sourceScores.getValue(RetrievalSource.VECTOR), 1e-6f)
        assertTrue(b.foundByBoth)

        val a = fused.single { it.chunk.id == "a" }
        assertEquals(setOf(RetrievalSource.VECTOR), a.ranks.keys)
        assertTrue(!a.foundByBoth)
    }

    @Test
    fun `a lower k trusts individual rankings more`() {
        // "top" is rank 1 in one source only; "mid" is rank 5 in both.
        val lists = listOf(
            list(RetrievalSource.VECTOR, 1.0, "top", "a", "b", "c", "mid"),
            list(RetrievalSource.KEYWORD, 1.0, "p", "q", "r", "s", "mid"),
        )

        val flat = ReciprocalRankFusion.fuse(lists, k = 60)
        val sharp = ReciprocalRankFusion.fuse(lists, k = 1)

        assertEquals("mid", flat.first().chunk.id, "at k=60 agreement across sources should win")
        assertEquals("top", sharp.first().chunk.id, "at k=1 a rank-1 position should dominate")
    }

    @Test
    fun `exact ties keep the order the sources produced`() {
        // a is (1st, 2nd), b is (2nd, 1st) — mathematically identical scores.
        // Unstable ordering here would make identical queries reshuffle results.
        val lists = listOf(
            list(RetrievalSource.VECTOR, 1.0, "a", "b"),
            list(RetrievalSource.KEYWORD, 1.0, "b", "a"),
        )

        val fused = ReciprocalRankFusion.fuse(lists)

        assertEquals(fused[0].score, fused[1].score, 1e-9f)
        assertEquals(listOf("a", "b"), fused.map { it.chunk.id })
        repeat(5) {
            assertEquals(listOf("a", "b"), ReciprocalRankFusion.fuse(lists).map { it.chunk.id })
        }
    }

    @Test
    fun `single source passes through in order`() {
        val fused = ReciprocalRankFusion.fuse(listOf(list(RetrievalSource.VECTOR, 1.0, "a", "b", "c")))
        assertEquals(listOf("a", "b", "c"), fused.map { it.chunk.id })
    }

    @Test
    fun `limit truncates after sorting not before`() {
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                list(RetrievalSource.VECTOR, 1.0, "a", "b", "c"),
                list(RetrievalSource.KEYWORD, 1.0, "c", "b"),
            ),
            limit = 1,
        )
        assertEquals(1, fused.size)
        assertEquals("c", fused.single().chunk.id)
    }

    @Test
    fun `empty and degenerate input is handled`() {
        assertTrue(ReciprocalRankFusion.fuse(emptyList()).isEmpty())
        assertTrue(ReciprocalRankFusion.fuse(listOf(list(RetrievalSource.VECTOR))).isEmpty())
        assertTrue(ReciprocalRankFusion.fuse(listOf(list(RetrievalSource.VECTOR, 1.0, "a")), limit = 0).isEmpty())
    }

    @Test
    fun `duplicate chunks across sources are merged not repeated`() {
        val fused = ReciprocalRankFusion.fuse(
            listOf(
                list(RetrievalSource.VECTOR, 1.0, "same"),
                list(RetrievalSource.KEYWORD, 1.0, "same"),
            ),
        )
        assertEquals(1, fused.size)
    }
}
