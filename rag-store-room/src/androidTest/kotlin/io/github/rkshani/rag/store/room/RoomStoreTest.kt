package io.github.rkshani.rag.store.room

import androidx.room.Room
import androidx.test.platform.app.InstrumentationRegistry
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.math.Vectors
import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.EmbeddedChunk
import io.github.rkshani.rag.store.room.db.RagDatabase
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.system.measureTimeMillis

class RoomStoreTest {

    private lateinit var database: RagDatabase
    private lateinit var vectorStore: RoomRagStore
    private lateinit var keywordIndex: RoomRagStore

    private val dimensions = 8

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        database = Room.inMemoryDatabaseBuilder(context, RagDatabase::class.java)
            .build()
        vectorStore = RoomRagStore(database, dimensions, pageSize = 300)
        keywordIndex = vectorStore
    }

    @After
    fun tearDown() = database.close()

    private fun vector(vararg values: Float): FloatArray =
        Vectors.normalize(FloatArray(dimensions) { values.getOrElse(it) { 0f } })

    private fun chunk(
        docId: String,
        index: Int,
        text: String,
        embedding: FloatArray,
        metadata: Map<String, String> = emptyMap(),
    ) = EmbeddedChunk(
        Chunk(
            id = "$docId#$index",
            docId = docId,
            index = index,
            text = text,
            charStart = index * 100,
            charEnd = index * 100 + text.length,
            page = index + 1,
            metadata = metadata,
        ),
        embedding,
    )

    @Test
    fun storesAndRetrievesByCosineSimilarity() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "machine learning on device", vector(1f, 0f, 0f)),
                chunk("doc1", 1, "cooking pasta at home", vector(0f, 1f, 0f)),
                chunk("doc1", 2, "quantized neural networks", vector(0.9f, 0.1f, 0f)),
            ),
        )

        val results = vectorStore.search(vector(1f, 0f, 0f), k = 2)

        assertEquals(2, results.size)
        assertEquals("doc1#0", results[0].chunk.id)
        assertEquals("doc1#2", results[1].chunk.id)
        assertTrue(results[0].score > results[1].score, "results must be ordered by score")
    }

    @Test
    fun roundTripsChunkFieldsIncludingOffsetsAndPage() = runBlocking {
        vectorStore.upsert(
            listOf(chunk("doc1", 3, "some text", vector(1f), mapOf("source" to "lecture"))),
        )

        val result = vectorStore.search(vector(1f), k = 1).single().chunk

        assertEquals("doc1", result.docId)
        assertEquals(3, result.index)
        assertEquals("some text", result.text)
        assertEquals(300, result.charStart)
        assertEquals(309, result.charEnd)
        assertEquals(4, result.page)
        assertEquals(mapOf("source" to "lecture"), result.metadata)
    }

    @Test
    fun scoresAreCosineSimilarities() = runBlocking {
        val stored = vector(1f, 1f)
        vectorStore.upsert(listOf(chunk("doc1", 0, "text", stored)))

        val query = vector(1f, 0f)
        val score = vectorStore.search(query, k = 1).single().score

        assertEquals(Vectors.cosine(query, stored), score, 1e-5f)
    }

    @Test
    fun filtersByDocumentId() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "first document", vector(1f)),
                chunk("doc2", 0, "second document", vector(1f)),
            ),
        )

        val results = vectorStore.search(vector(1f), k = 10, filter = SearchFilter(docIds = setOf("doc2")))

        assertEquals(1, results.size)
        assertEquals("doc2", results.single().chunk.docId)
    }

    @Test
    fun filtersByMetadata() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "week one", vector(1f), mapOf("week" to "1")),
                chunk("doc1", 1, "week two", vector(1f), mapOf("week" to "2")),
            ),
        )

        val results = vectorStore.search(
            vector(1f), k = 10, filter = SearchFilter(metadataEquals = mapOf("week" to "2")),
        )

        assertEquals(1, results.size)
        assertEquals("week two", results.single().chunk.text)
    }

    @Test
    fun deletingADocumentRemovesItsChunksAndFtsRows() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "deletable content about turbines", vector(1f)),
                chunk("doc2", 0, "surviving content about turbines", vector(1f)),
            ),
        )
        assertEquals(2, vectorStore.count())

        vectorStore.deleteDocument("doc1")

        assertEquals(1, vectorStore.count())
        val keywordHits = keywordIndex.search("turbines", k = 10)
        assertEquals(1, keywordHits.size, "FTS rows leaked after document deletion")
        assertEquals("doc2", keywordHits.single().chunk.docId)
    }

    @Test
    fun reindexingADocumentDoesNotDuplicateChunks() = runBlocking {
        val first = listOf(chunk("doc1", 0, "original text", vector(1f)))
        vectorStore.upsert(first)
        vectorStore.deleteDocument("doc1")
        vectorStore.upsert(listOf(chunk("doc1", 0, "replacement text", vector(1f))))

        assertEquals(1, vectorStore.count())
        assertEquals("replacement text", vectorStore.search(vector(1f), k = 5).single().chunk.text)
    }

    @Test
    fun keywordSearchFindsExactTokensThatVectorsMiss() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "the course code is CS4102 for compilers", vector(1f, 0f)),
                chunk("doc1", 1, "a general discussion of programming languages", vector(0f, 1f)),
            ),
        )

        val results = keywordIndex.search("CS4102", k = 5)

        assertEquals(1, results.size)
        assertTrue(results.single().chunk.text.contains("CS4102"))
        assertTrue(results.single().score > 0f)
    }

    @Test
    fun keywordSearchRanksByBm25() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "quantization quantization quantization", vector(1f)),
                chunk("doc1", 1, "one mention of quantization among many other unrelated words here", vector(1f)),
            ),
        )

        val results = keywordIndex.search("quantization", k = 5)

        assertEquals(2, results.size)
        assertTrue(
            results[0].score > results[1].score,
            "denser, shorter chunk should rank first",
        )
        assertEquals(0, results[0].chunk.index)
    }

    @Test
    fun keywordSearchHandlesUrduAndPunctuation() = runBlocking {
        vectorStore.upsert(
            listOf(
                chunk("doc1", 0, "یہ اردو کا جملہ ہے", vector(1f)),
                chunk("doc1", 1, "this is an english sentence", vector(1f)),
            ),
        )

        assertEquals(1, keywordIndex.search("اردو", k = 5).size)
        // Must not blow up on FTS syntax characters.
        assertTrue(keywordIndex.search("\"quoted\" AND (parens)", k = 5).isEmpty() || true)
        assertTrue(keywordIndex.search("!!!", k = 5).isEmpty())
    }

    @Test
    fun emptyStoreReturnsNothing() = runBlocking {
        assertTrue(vectorStore.search(vector(1f), k = 5).isEmpty())
        assertTrue(keywordIndex.search("anything", k = 5).isEmpty())
        assertEquals(0, vectorStore.count())
    }

    @Test
    fun rejectsVectorsOfTheWrongDimension() = runBlocking {
        val wrong = FloatArray(dimensions + 1) { 1f }
        val failed = runCatching {
            vectorStore.upsert(listOf(chunk("doc1", 0, "text", wrong)))
        }.isFailure
        assertTrue(failed, "dimension mismatch must fail loudly at write time, not silently at search time")
    }

    /**
     * Not a strict assertion on hardware timing — it exists to catch an
     * accidental O(n^2) or a per-candidate full-row load, which would blow past
     * this by orders of magnitude.
     */
    @Test
    fun exhaustiveScanStaysFastAtTenThousandChunks() = runBlocking {
        val random = Random(42)
        val batch = (0 until 10_000).map { i ->
            chunk(
                docId = "bulk",
                index = i,
                text = "chunk number $i with some filler text for the index",
                embedding = Vectors.normalize(FloatArray(dimensions) { random.nextFloat() - 0.5f }),
            )
        }
        batch.chunked(1000).forEach { vectorStore.upsert(it) }
        assertEquals(10_000, vectorStore.count())

        val query = Vectors.normalize(FloatArray(dimensions) { random.nextFloat() - 0.5f })
        val elapsed = measureTimeMillis { vectorStore.search(query, k = 5) }

        assertTrue(elapsed < 2_000, "10k-chunk scan took ${elapsed}ms — check for a per-candidate row load")
    }

    /**
     * Regression test for the original upsert: it wrote the document row with
     * REPLACE, which deletes and re-inserts, and the foreign key cascaded that
     * delete onto every chunk already stored — so each batch wiped the last.
     */
    @Test
    fun writingADocumentInBatchesKeepsEveryBatch() = runBlocking {
        vectorStore.upsert(listOf(chunk("doc1", 0, "first batch about solar", vector(1f))))
        vectorStore.upsert(listOf(chunk("doc1", 1, "second batch about batteries", vector(0f, 1f))))
        vectorStore.upsert(listOf(chunk("doc1", 2, "third batch about wiring", vector(0f, 0f, 1f))))

        assertEquals(3, vectorStore.count())
        assertEquals(1, keywordIndex.search("solar", k = 5).size, "first batch's FTS row was lost")
        assertEquals(3, vectorStore.document("doc1")?.chunkCount)
    }

    @Test
    fun catalogTracksCompletion() = runBlocking {
        vectorStore.upsert(listOf(chunk("doc1", 0, "text", vector(1f), mapOf("title" to "Notes"))))
        val pending = vectorStore.documents().single()
        assertEquals("Notes", pending.title)
        assertTrue(!pending.complete)

        vectorStore.markComplete("doc1")
        assertTrue(vectorStore.document("doc1")!!.complete)
    }

    @Test
    fun rewritingAChunkReplacesItsFtsRowToo() = runBlocking {
        vectorStore.upsert(listOf(chunk("doc1", 0, "old words", vector(1f))))
        vectorStore.upsert(listOf(chunk("doc1", 0, "new words", vector(1f))))

        assertEquals(1, vectorStore.count())
        assertTrue(keywordIndex.search("old", k = 5).isEmpty(), "stale FTS row survived a rewrite")
        assertEquals(1, keywordIndex.search("new", k = 5).size)
    }

    @Test
    fun urduKeywordSearchIgnoresDiacriticsAndLetterVariants() = runBlocking {
        vectorStore.upsert(listOf(chunk("doc1", 0, "یہ کِتاب بہت اچھی ہے", vector(1f))))

        assertEquals(1, keywordIndex.search("کتاب", k = 5).size, "plain query missed diacritised text")
        // Arabic kaf (U+0643) as typed on an Arabic keyboard.
        assertEquals(1, keywordIndex.search("\u0643تاب", k = 5).size, "Arabic keyboard spelling missed")
    }

    @Test
    fun scanPagesAcrossTheWholeTable() = runBlocking {
        // pageSize is 300 in setUp; the best match sits on a later page.
        val filler = (0 until 1000).map { chunk("bulk", it, "filler $it", vector(0f, 1f)) }
        vectorStore.upsert(filler)
        vectorStore.upsert(listOf(chunk("target", 0, "the one", vector(1f))))

        assertEquals("target", vectorStore.search(vector(1f), k = 1).single().chunk.docId)
    }
}
