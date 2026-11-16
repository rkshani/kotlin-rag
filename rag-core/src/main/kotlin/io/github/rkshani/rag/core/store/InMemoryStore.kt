package io.github.rkshani.rag.core.store

import io.github.rkshani.rag.core.DocumentCatalog
import io.github.rkshani.rag.core.DocumentInfo
import io.github.rkshani.rag.core.KeywordIndex
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.VectorStore
import io.github.rkshani.rag.core.math.Vectors
import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.model.EmbeddedChunk
import io.github.rkshani.rag.core.model.ScoredChunk
import io.github.rkshani.rag.core.text.SearchText
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.PriorityQueue
import kotlin.math.ln

/**
 * Vector store, BM25 keyword index and document catalog in one in-memory object.
 *
 * For tests, JVM use, and apps whose corpus is small enough to rebuild on launch.
 * Nothing persists. Pass the same instance as the vector store, keyword index and
 * catalog. All operations are serialised by one mutex.
 */
class InMemoryStore(
    override val dimensions: Int? = null,
    private val clock: () -> Long = System::currentTimeMillis,
    private val k1: Double = 1.2,
    private val b: Double = 0.75,
) : VectorStore, KeywordIndex, DocumentCatalog {

    private class Entry(val chunk: Chunk, val vector: FloatArray, val terms: Map<String, Int>, val length: Int)

    private class Doc(val docId: String, val title: String?, val metadata: Map<String, String>, var indexedAt: Long, var complete: Boolean)

    private val mutex = Mutex()
    private val entries = LinkedHashMap<String, Entry>()
    private val docs = LinkedHashMap<String, Doc>()
    private val postings = HashMap<String, MutableSet<String>>()
    private var totalLength = 0L

    override suspend fun upsert(chunks: List<EmbeddedChunk>) = mutex.withLock {
        chunks.forEach { embedded ->
            dimensions?.let {
                require(embedded.embedding.size == it) {
                    "chunk ${embedded.chunk.id} has ${embedded.embedding.size} dims, store expects $it"
                }
            }
            val chunk = embedded.chunk
            removeEntry(chunk.id)
            val tokens = tokenize(chunk.text)
            val terms = tokens.groupingBy { it }.eachCount()
            entries[chunk.id] = Entry(chunk, embedded.embedding.copyOf(), terms, tokens.size)
            terms.keys.forEach { postings.getOrPut(it) { HashSet() }.add(chunk.id) }
            totalLength += tokens.size
            docs.getOrPut(chunk.docId) {
                Doc(chunk.docId, chunk.metadata["title"], chunk.metadata, clock(), complete = false)
            }
        }
    }

    /** Text is indexed in [upsert]; nothing more to do here. */
    override suspend fun index(chunks: List<EmbeddedChunk>) = Unit

    override suspend fun search(query: FloatArray, k: Int, filter: SearchFilter?): List<ScoredChunk> {
        if (k <= 0) return emptyList()
        val snapshot = mutex.withLock { entries.values.toList() }
        val best = PriorityQueue<ScoredChunk>(k) { a, c -> a.score.compareTo(c.score) }
        snapshot.forEachIndexed { i, entry ->
            if (i % 1024 == 0) currentCoroutineContext().ensureActive()
            if (!matches(entry.chunk, filter) || entry.vector.size != query.size) return@forEachIndexed
            offer(best, k, ScoredChunk(entry.chunk, Vectors.dot(query, entry.vector)))
        }
        return best.sortedByDescending { it.score }
    }

    override suspend fun search(query: String, k: Int, filter: SearchFilter?): List<ScoredChunk> {
        if (k <= 0) return emptyList()
        val queryTerms = tokenize(query).distinct()
        if (queryTerms.isEmpty()) return emptyList()
        return mutex.withLock {
            val n = entries.size
            if (n == 0) return@withLock emptyList()
            val avgLength = (totalLength.toDouble() / n).coerceAtLeast(1.0)
            val scores = HashMap<String, Double>()
            for (term in queryTerms) {
                val ids = postings[term] ?: continue
                val idf = ln(1.0 + (n - ids.size + 0.5) / (ids.size + 0.5))
                for (id in ids) {
                    val entry = entries.getValue(id)
                    val tf = entry.terms.getValue(term).toDouble()
                    val norm = tf + k1 * (1 - b + b * entry.length / avgLength)
                    scores[id] = (scores[id] ?: 0.0) + idf * tf * (k1 + 1) / norm
                }
            }
            val best = PriorityQueue<ScoredChunk>(k) { a, c -> a.score.compareTo(c.score) }
            scores.forEach { (id, score) ->
                val chunk = entries.getValue(id).chunk
                if (matches(chunk, filter)) offer(best, k, ScoredChunk(chunk, score.toFloat()))
            }
            best.sortedByDescending { it.score }
        }
    }

    override suspend fun deleteDocument(docId: String) = mutex.withLock {
        entries.values.filter { it.chunk.docId == docId }.map { it.chunk.id }.forEach(::removeEntry)
        docs.remove(docId)
        Unit
    }

    override suspend fun count(): Int = mutex.withLock { entries.size }

    override suspend fun documents(): List<DocumentInfo> = mutex.withLock { docs.values.map(::info) }

    override suspend fun document(docId: String): DocumentInfo? = mutex.withLock { docs[docId]?.let(::info) }

    override suspend fun markComplete(docId: String) = mutex.withLock {
        docs[docId]?.complete = true
        Unit
    }

    private fun info(doc: Doc) = DocumentInfo(
        docId = doc.docId,
        title = doc.title,
        chunkCount = entries.values.count { it.chunk.docId == doc.docId },
        indexedAtMillis = doc.indexedAt,
        complete = doc.complete,
        metadata = doc.metadata,
    )

    private fun removeEntry(id: String) {
        val old = entries.remove(id) ?: return
        totalLength -= old.length
        old.terms.keys.forEach { term ->
            postings[term]?.let { set ->
                set.remove(id)
                if (set.isEmpty()) postings.remove(term)
            }
        }
    }

    private fun matches(chunk: Chunk, filter: SearchFilter?): Boolean {
        filter ?: return true
        if (filter.docIds != null && chunk.docId !in filter.docIds) return false
        return filter.metadataEquals.all { (key, value) -> chunk.metadata[key] == value }
    }

    private fun offer(queue: PriorityQueue<ScoredChunk>, k: Int, item: ScoredChunk) {
        if (queue.size < k) {
            queue.add(item)
        } else if (item.score > queue.peek().score) {
            queue.poll()
            queue.add(item)
        }
    }

    private fun tokenize(text: String): List<String> =
        SearchText.normalize(text).split(Regex("""[^\p{L}\p{N}\p{M}]+""")).filter { it.isNotEmpty() }
}
