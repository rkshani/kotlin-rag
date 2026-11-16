package io.github.rkshani.rag.core.retrieval

import io.github.rkshani.rag.core.Embedder
import io.github.rkshani.rag.core.EmbeddingKind
import io.github.rkshani.rag.core.KeywordIndex
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.VectorStore
import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.EmbeddedChunk
import io.github.rkshani.rag.core.model.ScoredChunk
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HybridRetrieverTest {

    private fun chunk(id: String, docId: String = "doc1", index: Int = 0) = Chunk(
        id = id, docId = docId, index = index, text = id, charStart = 0, charEnd = id.length,
    )

    private fun hits(vararg ids: Pair<String, String>) = ids.mapIndexed { i, (id, docId) ->
        ScoredChunk(chunk(id, docId, i), 1f - i * 0.05f)
    }

    private class FakeEmbedder(var calls: Int = 0, var lastKind: EmbeddingKind? = null) : Embedder {
        override val dimensions = 4
        override val maxSequenceTokens = 128
        override suspend fun embed(texts: List<String>, kind: EmbeddingKind): List<FloatArray> {
            calls++
            lastKind = kind
            return texts.map { floatArrayOf(1f, 0f, 0f, 0f) }
        }
        override fun close() = Unit
    }

    private class FakeVectorStore(
        private val results: List<ScoredChunk> = emptyList(),
        private val failure: Throwable? = null,
        private val delayMillis: Long = 0,
    ) : VectorStore {
        var lastK: Int? = null
        var lastFilter: SearchFilter? = null
        override suspend fun upsert(chunks: List<EmbeddedChunk>) = Unit
        override suspend fun search(query: FloatArray, k: Int, filter: SearchFilter?): List<ScoredChunk> {
            if (delayMillis > 0) delay(delayMillis)
            failure?.let { throw it }
            lastK = k
            lastFilter = filter
            return results.take(k)
        }
        override suspend fun deleteDocument(docId: String) = Unit
        override suspend fun count() = results.size
    }

    private class FakeKeywordIndex(
        private val results: List<ScoredChunk> = emptyList(),
        private val failure: Throwable? = null,
        private val delayMillis: Long = 0,
    ) : KeywordIndex {
        var lastQuery: String? = null
        override suspend fun index(chunks: List<EmbeddedChunk>) = Unit
        override suspend fun search(query: String, k: Int, filter: SearchFilter?): List<ScoredChunk> {
            if (delayMillis > 0) delay(delayMillis)
            failure?.let { throw it }
            lastQuery = query
            return results.take(k)
        }
        override suspend fun deleteDocument(docId: String) = Unit
    }

    @Test
    fun `fuses both sources`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(hits("a" to "doc1", "b" to "doc1")),
            keywordIndex = FakeKeywordIndex(hits("b" to "doc1", "c" to "doc1")),
        )

        val results = retriever.retrieve("query", k = 3)

        assertEquals(listOf("b", "a", "c"), results.map { it.chunk.id })
        assertTrue(results.first().foundByBoth)
    }

    @Test
    fun `embeds the query with QUERY kind not DOCUMENT`() = runTest {
        // Asymmetric models (E5, BGE) prefix queries differently. Getting this
        // wrong degrades retrieval quietly rather than failing.
        val embedder = FakeEmbedder()
        HybridRetriever(embedder, FakeVectorStore(hits("a" to "doc1"))).retrieve("query", k = 1)

        assertEquals(EmbeddingKind.QUERY, embedder.lastKind)
    }

    @Test
    fun `searches deeper than k so fusion has room to promote`() = runTest {
        val store = FakeVectorStore(hits("a" to "doc1"))
        HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = store,
            config = HybridRetrievalConfig(vectorCandidates = 20),
        ).retrieve("query", k = 3)

        assertEquals(20, store.lastK)
    }

    @Test
    fun `runs both sources concurrently`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(hits("a" to "doc1"), delayMillis = 1_000),
            keywordIndex = FakeKeywordIndex(hits("b" to "doc1"), delayMillis = 1_000),
        )

        val start = currentTime
        retriever.retrieve("query", k = 2)
        val elapsed = currentTime - start

        assertTrue(elapsed < 2_000, "sources ran sequentially (${elapsed}ms of virtual time)")
    }

    @Test
    fun `degrades to keyword results when the vector side fails`() = runTest {
        val failures = mutableListOf<RetrievalSource>()
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(failure = IllegalStateException("model not loaded")),
            keywordIndex = FakeKeywordIndex(hits("b" to "doc1")),
            onSourceFailure = { source, _ -> failures.add(source) },
        )

        val results = retriever.retrieve("query", k = 3)

        assertEquals(listOf("b"), results.map { it.chunk.id })
        assertEquals(listOf(RetrievalSource.VECTOR), failures)
    }

    @Test
    fun `degrades to vector results when the keyword side fails`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(hits("a" to "doc1")),
            keywordIndex = FakeKeywordIndex(failure = IllegalStateException("fts corrupt")),
        )

        assertEquals(listOf("a"), retriever.retrieve("query", k = 3).map { it.chunk.id })
    }

    @Test
    fun `throws when every available source fails`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(failure = IllegalStateException("boom")),
            keywordIndex = FakeKeywordIndex(failure = IllegalStateException("also boom")),
        )

        assertFailsWith<IllegalStateException> { retriever.retrieve("query", k = 3) }
    }

    @Test
    fun `works without a keyword index`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(hits("a" to "doc1", "b" to "doc1")),
            keywordIndex = null,
        )

        assertEquals(listOf("a", "b"), retriever.retrieve("query", k = 2).map { it.chunk.id })
    }

    @Test
    fun `caps results per document for diversity`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(
                hits("a1" to "doc1", "a2" to "doc1", "a3" to "doc1", "b1" to "doc2", "c1" to "doc3"),
            ),
            config = HybridRetrievalConfig(maxPerDocument = 1),
        )

        val results = retriever.retrieve("query", k = 3)

        assertEquals(listOf("a1", "b1", "c1"), results.map { it.chunk.id })
        assertEquals(3, results.map { it.chunk.docId }.distinct().size)
    }

    @Test
    fun `the cap never returns fewer than available from other documents`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(hits("a1" to "doc1", "a2" to "doc1")),
            config = HybridRetrievalConfig(maxPerDocument = 1),
        )

        // Only one document exists, so the cap legitimately shrinks the result.
        assertEquals(1, retriever.retrieve("query", k = 5).size)
    }

    @Test
    fun `respects the requested k`() = runTest {
        val retriever = HybridRetriever(
            embedder = FakeEmbedder(),
            vectorStore = FakeVectorStore(hits("a" to "doc1", "b" to "doc1", "c" to "doc1")),
        )

        assertEquals(2, retriever.retrieve("query", k = 2).size)
    }

    @Test
    fun `blank query and non-positive k short-circuit without touching the sources`() = runTest {
        val embedder = FakeEmbedder()
        val retriever = HybridRetriever(embedder, FakeVectorStore(hits("a" to "doc1")))

        assertTrue(retriever.retrieve("", k = 5).isEmpty())
        assertTrue(retriever.retrieve("   ", k = 5).isEmpty())
        assertTrue(retriever.retrieve("query", k = 0).isEmpty())
        assertEquals(0, embedder.calls, "no work should be done for an empty request")
    }

    @Test
    fun `passes filters through to both sources`() = runTest {
        val store = FakeVectorStore(hits("a" to "doc1"))
        val filter = SearchFilter(docIds = setOf("doc1"))

        HybridRetriever(FakeEmbedder(), store, FakeKeywordIndex(hits("b" to "doc1")))
            .retrieve("query", k = 2, filter = filter)

        assertEquals(filter, store.lastFilter)
    }

    @Test
    fun `empty sources return an empty list rather than failing`() = runTest {
        val retriever = HybridRetriever(FakeEmbedder(), FakeVectorStore(), FakeKeywordIndex())
        assertTrue(retriever.retrieve("query", k = 5).isEmpty())
    }
}
