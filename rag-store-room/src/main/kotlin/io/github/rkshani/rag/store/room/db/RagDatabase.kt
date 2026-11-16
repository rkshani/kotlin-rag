package io.github.rkshani.rag.store.room.db

import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Fts4
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import androidx.room.Transaction

@Entity(tableName = "documents")
data class DocumentEntity(
    @PrimaryKey val docId: String,
    val title: String?,
    val metadataJson: String,
    val indexedAt: Long,
    /** Set once every chunk is stored. Stays false if the process died mid-index. */
    val complete: Boolean = false,
)

@Entity(
    tableName = "chunks",
    indices = [Index("docId"), Index(value = ["docId", "chunkIndex"], unique = true)],
    foreignKeys = [
        ForeignKey(
            entity = DocumentEntity::class,
            parentColumns = ["docId"],
            childColumns = ["docId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
)
data class ChunkEntity(
    @PrimaryKey(autoGenerate = true) val rowId: Long = 0,
    val docId: String,
    val chunkIndex: Int,
    val text: String,
    val charStart: Int,
    val charEnd: Int,
    val page: Int?,
    val metadataJson: String,
    /** Float32 little-endian, L2-normalised. See Vectors.toBytes. */
    val embedding: ByteArray,
    val dimensions: Int,
) {
    // ByteArray in a data class gives identity equals; Room does not care, but a
    // silently wrong equals() causes very confusing test failures.
    override fun equals(other: Any?): Boolean = this === other || (other is ChunkEntity && rowId == other.rowId)
    override fun hashCode(): Int = rowId.hashCode()
}

/**
 * Standalone FTS4 table holding the chunk text in its search form (see
 * `SearchText.normalize`): Urdu diacritics removed, Arabic letter variants
 * folded, lowercased. The display text stays untouched in [ChunkEntity].
 *
 * An external-content table would avoid the second copy, but Room does not
 * generate its sync triggers, so every write path would have to maintain the
 * index by hand. A 384-dim vector is 1536 bytes — already more than a typical
 * chunk's text — so the copy costs well under 50% extra and buys correctness.
 */
@Fts4(tokenizer = "unicode61")
@Entity(tableName = "chunks_fts")
data class ChunkFtsEntity(
    @PrimaryKey @ColumnInfo(name = "rowid") val rowId: Long,
    val text: String,
)

/** Projection for keyword search: the chunk plus its raw ranking statistics. */
data class ChunkWithMatchInfo(
    val rowId: Long,
    val docId: String,
    val chunkIndex: Int,
    val text: String,
    val charStart: Int,
    val charEnd: Int,
    val page: Int?,
    val metadataJson: String,
    val matchInfo: ByteArray,
) {
    override fun equals(other: Any?): Boolean = this === other || (other is ChunkWithMatchInfo && rowId == other.rowId)
    override fun hashCode(): Int = rowId.hashCode()
}

/** Projection for vector search — text is not loaded until a chunk survives top-K. */
data class ChunkVector(
    val rowId: Long,
    val docId: String,
    val metadataJson: String,
    val embedding: ByteArray,
) {
    override fun equals(other: Any?): Boolean = this === other || (other is ChunkVector && rowId == other.rowId)
    override fun hashCode(): Int = rowId.hashCode()
}

data class DocumentRow(
    val docId: String,
    val title: String?,
    val metadataJson: String,
    val indexedAt: Long,
    val complete: Boolean,
    val chunkCount: Int,
)

/**
 * Abstract class rather than interface: Room only wraps a DAO's own method
 * bodies in [Transaction] reliably for abstract classes.
 */
@Dao
abstract class RagDao {

    /**
     * IGNORE, never REPLACE. SQLite's REPLACE deletes the existing row before
     * inserting, and the chunk foreign key cascades that delete — so writing a
     * document's second batch would wipe its first batch's chunks, and leave
     * their FTS rows orphaned.
     */
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    abstract suspend fun insertDocumentIfAbsent(document: DocumentEntity): Long

    @Query("UPDATE documents SET complete = 1 WHERE docId = :docId")
    abstract suspend fun markComplete(docId: String)

    @Insert(onConflict = OnConflictStrategy.ABORT)
    abstract suspend fun insertChunks(chunks: List<ChunkEntity>): List<Long>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    abstract suspend fun insertFts(rows: List<ChunkFtsEntity>)

    @Query("DELETE FROM chunks_fts WHERE rowid IN (SELECT rowId FROM chunks WHERE docId = :docId AND chunkIndex IN (:indices))")
    abstract suspend fun deleteFtsForChunks(docId: String, indices: List<Int>)

    @Query("DELETE FROM chunks WHERE docId = :docId AND chunkIndex IN (:indices)")
    abstract suspend fun deleteChunks(docId: String, indices: List<Int>)

    /**
     * Writes a batch of one document's chunks and their FTS rows atomically.
     * Re-writing an existing (docId, chunkIndex) replaces it explicitly, FTS
     * row first, so nothing is orphaned.
     */
    @Transaction
    open suspend fun upsertChunks(document: DocumentEntity, chunks: List<ChunkEntity>, searchTexts: List<String>) {
        require(chunks.size == searchTexts.size) { "one search text per chunk" }
        insertDocumentIfAbsent(document)
        val indices = chunks.map { it.chunkIndex }
        deleteFtsForChunks(document.docId, indices)
        deleteChunks(document.docId, indices)
        val ids = insertChunks(chunks)
        insertFts(ids.mapIndexed { i, id -> ChunkFtsEntity(rowId = id, text = searchTexts[i]) })
    }

    /** Keyset paging keeps memory flat: a 100k-chunk index is never loaded at once. */
    @Query("SELECT rowId, docId, metadataJson, embedding FROM chunks WHERE rowId > :afterRowId ORDER BY rowId LIMIT :limit")
    abstract suspend fun vectorPage(afterRowId: Long, limit: Int): List<ChunkVector>

    @Query(
        "SELECT rowId, docId, metadataJson, embedding FROM chunks " +
            "WHERE docId IN (:docIds) AND rowId > :afterRowId ORDER BY rowId LIMIT :limit",
    )
    abstract suspend fun vectorPageForDocuments(docIds: List<String>, afterRowId: Long, limit: Int): List<ChunkVector>

    @Query("SELECT * FROM chunks WHERE rowId IN (:rowIds)")
    abstract suspend fun chunksByRowIds(rowIds: List<Long>): List<ChunkEntity>

    /**
     * FTS4 candidate fetch. Ranking happens in Kotlin — see Bm25.
     * [limit] caps how many rows we score; FTS4 returns them in rowid order,
     * not relevance order, so this is a recall knob rather than a top-K.
     */
    @Query(
        """
        SELECT c.rowId AS rowId, c.docId AS docId, c.chunkIndex AS chunkIndex, c.text AS text,
               c.charStart AS charStart, c.charEnd AS charEnd, c.page AS page,
               c.metadataJson AS metadataJson,
               matchinfo(chunks_fts, 'pcnalx') AS matchInfo
        FROM chunks_fts
        JOIN chunks c ON c.rowId = chunks_fts.rowid
        WHERE chunks_fts MATCH :query
        LIMIT :limit
        """,
    )
    abstract suspend fun keywordSearch(query: String, limit: Int): List<ChunkWithMatchInfo>

    @Query("DELETE FROM chunks_fts WHERE rowid IN (SELECT rowId FROM chunks WHERE docId = :docId)")
    abstract suspend fun deleteFtsForDocument(docId: String)

    @Query("DELETE FROM documents WHERE docId = :docId")
    abstract suspend fun deleteDocumentRow(docId: String)

    /** FTS rows must go first — the chunk rows they reference are about to cascade away. */
    @Transaction
    open suspend fun deleteDocument(docId: String) {
        deleteFtsForDocument(docId)
        deleteDocumentRow(docId)
    }

    @Query("SELECT COUNT(*) FROM chunks")
    abstract suspend fun chunkCount(): Int

    @Query(
        """
        SELECT d.docId AS docId, d.title AS title, d.metadataJson AS metadataJson, d.indexedAt AS indexedAt,
               d.complete AS complete,
               (SELECT COUNT(*) FROM chunks c WHERE c.docId = d.docId) AS chunkCount
        FROM documents d
        ORDER BY d.indexedAt DESC
        """,
    )
    abstract suspend fun documents(): List<DocumentRow>

    @Query(
        """
        SELECT d.docId AS docId, d.title AS title, d.metadataJson AS metadataJson, d.indexedAt AS indexedAt,
               d.complete AS complete,
               (SELECT COUNT(*) FROM chunks c WHERE c.docId = d.docId) AS chunkCount
        FROM documents d WHERE d.docId = :docId
        """,
    )
    abstract suspend fun document(docId: String): DocumentRow?
}

@Database(
    entities = [DocumentEntity::class, ChunkEntity::class, ChunkFtsEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class RagDatabase : RoomDatabase() {
    abstract fun ragDao(): RagDao

    companion object {
        const val NAME: String = "kotlin-rag.db"
    }
}
