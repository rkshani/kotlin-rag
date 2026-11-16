package io.github.rkshani.rag.store.room

import android.content.Context
import androidx.room.Room
import io.github.rkshani.rag.core.DocumentCatalog
import io.github.rkshani.rag.core.DocumentInfo
import io.github.rkshani.rag.core.KeywordIndex
import io.github.rkshani.rag.core.RagDispatchers
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.VectorStore
import io.github.rkshani.rag.core.math.Vectors
import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.EmbeddedChunk
import io.github.rkshani.rag.core.model.ScoredChunk
import io.github.rkshani.rag.core.text.SearchText
import io.github.rkshani.rag.store.room.db.ChunkEntity
import io.github.rkshani.rag.store.room.db.DocumentEntity
import io.github.rkshani.rag.store.room.db.DocumentRow
import io.github.rkshani.rag.store.room.db.RagDao
import io.github.rkshani.rag.store.room.db.RagDatabase
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Vector store, BM25 keyword index and document catalog over one Room database.
 *
 * Pass it to `KotlinRag.Builder.store(...)` — it plays all three roles.
 *
 * Vector search is an exhaustive scan, which sounds naive and is the right call
 * at this scale: 10k chunks x 384 dims is ~4M multiply-adds, a few milliseconds,
 * with exact results and no index to build, tune or invalidate on insert. The
 * scan pages through the table so memory stays flat as the index grows, and
 * checks for cancellation between pages. Revisit above ~100k chunks.
 */
class RoomRagStore(
    private val database: RagDatabase,
    override val dimensions: Int,
    /** How many FTS rows to score per keyword query. Higher costs time, lower costs recall. */
    private val keywordCandidates: Int = 200,
    private val pageSize: Int = 2_000,
    private val dispatchers: RagDispatchers = RagDispatchers(),
    private val clock: () -> Long = System::currentTimeMillis,
) : VectorStore, KeywordIndex, DocumentCatalog, AutoCloseable {

    private val dao: RagDao = database.ragDao()

    override suspend fun upsert(chunks: List<EmbeddedChunk>) {
        if (chunks.isEmpty()) return
        chunks.forEach {
            require(it.embedding.size == dimensions) {
                "chunk ${it.chunk.id} has ${it.embedding.size} dims, store expects $dimensions"
            }
        }
        withContext(dispatchers.io) {
            chunks.groupBy { it.chunk.docId }.forEach { (docId, forDocument) ->
                val first = forDocument.first().chunk
                dao.upsertChunks(
                    document = DocumentEntity(
                        docId = docId,
                        title = first.metadata["title"],
                        metadataJson = MetadataJson.encode(first.metadata),
                        indexedAt = clock(),
                    ),
                    chunks = forDocument.map { it.toEntity() },
                    searchTexts = forDocument.map { SearchText.normalize(it.chunk.text) },
                )
            }
        }
    }

    /** FTS rows are written by [upsert] in the same transaction. */
    override suspend fun index(chunks: List<EmbeddedChunk>) = Unit

    override suspend fun search(query: FloatArray, k: Int, filter: SearchFilter?): List<ScoredChunk> {
        require(query.size == dimensions) { "query has ${query.size} dims, store holds $dimensions" }
        if (k <= 0) return emptyList()
        val docIds = filter?.docIds?.toList()
        if (docIds != null && docIds.isEmpty()) return emptyList()
        val metadataFilter = filter?.metadataEquals.orEmpty()

        return withContext(dispatchers.default) {
            val topK = TopK<Pair<Long, Float>>(k) { it.second }
            var after = 0L
            while (true) {
                currentCoroutineContext().ensureActive()
                val page = if (docIds == null) {
                    dao.vectorPage(after, pageSize)
                } else {
                    dao.vectorPageForDocuments(docIds, after, pageSize)
                }
                if (page.isEmpty()) break
                page.forEach { candidate ->
                    // docId filtering is pushed into SQL; metadata lives in JSON. If
                    // metadata filters become hot, promote them to real columns.
                    if (MetadataJson.matches(candidate.metadataJson, metadataFilter)) {
                        topK.offer(candidate.rowId to dotProduct(query, candidate.embedding))
                    }
                }
                after = page.last().rowId
                if (page.size < pageSize) break
            }

            val ranked = topK.drain()
            if (ranked.isEmpty()) return@withContext emptyList()
            val byRowId = dao.chunksByRowIds(ranked.map { it.first }).associateBy { it.rowId }
            ranked.mapNotNull { (rowId, score) -> byRowId[rowId]?.let { ScoredChunk(it.toChunk(), score) } }
        }
    }

    /**
     * Lexical retrieval over FTS4, ranked with [Bm25]. Query and indexed text go
     * through the same normalisation, so "کتاب" finds "کِتاب" and an
     * Arabic-keyboard "كتاب" finds both.
     */
    override suspend fun search(query: String, k: Int, filter: SearchFilter?): List<ScoredChunk> {
        if (k <= 0) return emptyList()
        val expression = FtsQuery.sanitize(SearchText.normalize(query)) ?: return emptyList()
        val docIds = filter?.docIds
        val metadataFilter = filter?.metadataEquals.orEmpty()

        return withContext(dispatchers.default) {
            val rows = dao.keywordSearch(expression, keywordCandidates)
            val topK = TopK<ScoredChunk>(k) { it.score }
            rows.forEach { row ->
                if (docIds != null && row.docId !in docIds) return@forEach
                if (!MetadataJson.matches(row.metadataJson, metadataFilter)) return@forEach
                val score = Bm25.score(row.matchInfo).toFloat()
                if (score <= 0f) return@forEach
                topK.offer(
                    ScoredChunk(
                        chunk = Chunk(
                            id = "${row.docId}#${row.chunkIndex}",
                            docId = row.docId,
                            index = row.chunkIndex,
                            text = row.text,
                            charStart = row.charStart,
                            charEnd = row.charEnd,
                            page = row.page,
                            metadata = MetadataJson.decode(row.metadataJson),
                        ),
                        score = score,
                    ),
                )
            }
            topK.drain()
        }
    }

    override suspend fun deleteDocument(docId: String) = withContext(dispatchers.io) { dao.deleteDocument(docId) }

    override suspend fun count(): Int = withContext(dispatchers.io) { dao.chunkCount() }

    override suspend fun documents(): List<DocumentInfo> = withContext(dispatchers.io) { dao.documents().map { it.toInfo() } }

    override suspend fun document(docId: String): DocumentInfo? = withContext(dispatchers.io) { dao.document(docId)?.toInfo() }

    override suspend fun markComplete(docId: String) = withContext(dispatchers.io) { dao.markComplete(docId) }

    override fun close() = database.close()

    /** Reads the float32 BLOB in place instead of allocating a FloatArray per candidate. */
    private fun dotProduct(query: FloatArray, blob: ByteArray): Float {
        val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        if (buffer.limit() != query.size) return -1f
        var sum = 0f
        for (i in query.indices) sum += query[i] * buffer.get(i)
        return sum
    }

    companion object {
        /**
         * Opens (or creates) the database file. One file per embedding model:
         * vectors from different models are not comparable, so switching models
         * means a new [name] and a re-index.
         */
        fun create(
            context: Context,
            dimensions: Int,
            name: String = RagDatabase.NAME,
            dispatchers: RagDispatchers = RagDispatchers(),
        ): RoomRagStore {
            val database = Room.databaseBuilder(context.applicationContext, RagDatabase::class.java, name).build()
            return RoomRagStore(database, dimensions, dispatchers = dispatchers)
        }
    }
}

private fun DocumentRow.toInfo() = DocumentInfo(
    docId = docId,
    title = title,
    chunkCount = chunkCount,
    indexedAtMillis = indexedAt,
    complete = complete,
    metadata = MetadataJson.decode(metadataJson),
)

internal fun EmbeddedChunk.toEntity(): ChunkEntity = ChunkEntity(
    docId = chunk.docId,
    chunkIndex = chunk.index,
    text = chunk.text,
    charStart = chunk.charStart,
    charEnd = chunk.charEnd,
    page = chunk.page,
    metadataJson = MetadataJson.encode(chunk.metadata),
    embedding = Vectors.toBytes(embedding),
    dimensions = embedding.size,
)

internal fun ChunkEntity.toChunk(): Chunk = Chunk(
    id = "$docId#$chunkIndex",
    docId = docId,
    index = chunkIndex,
    text = text,
    charStart = charStart,
    charEnd = charEnd,
    page = page,
    metadata = MetadataJson.decode(metadataJson),
)
