package io.github.rkshani.rag.store.room

import org.json.JSONObject
import java.util.PriorityQueue

/**
 * Keeps only the K best items seen.
 *
 * Sorting all N candidates to take 5 allocates a full list and costs O(N log N);
 * a bounded min-heap is O(N log K) and holds K entries. At 100k chunks that is
 * the difference between a visible pause and an imperceptible one.
 */
internal class TopK<T>(private val k: Int, private val scoreOf: (T) -> Float) {

    private val heap = PriorityQueue<T>(k.coerceAtLeast(1)) { a, b ->
        scoreOf(a).compareTo(scoreOf(b))
    }

    fun offer(item: T) {
        if (heap.size < k) {
            heap.add(item)
        } else if (scoreOf(item) > scoreOf(heap.peek()!!)) {
            heap.poll()
            heap.add(item)
        }
    }

    /** Highest score first. */
    fun drain(): List<T> = heap.sortedByDescending(scoreOf)
}

internal object MetadataJson {

    fun encode(metadata: Map<String, String>): String {
        if (metadata.isEmpty()) return "{}"
        val json = JSONObject()
        metadata.forEach { (key, value) -> json.put(key, value) }
        return json.toString()
    }

    fun decode(json: String): Map<String, String> {
        if (json.isBlank() || json == "{}") return emptyMap()
        return runCatching {
            val obj = JSONObject(json)
            buildMap {
                obj.keys().forEach { key -> put(key, obj.optString(key)) }
            }
        }.getOrDefault(emptyMap())
    }

    fun matches(json: String, required: Map<String, String>): Boolean {
        if (required.isEmpty()) return true
        val actual = decode(json)
        return required.all { (key, value) -> actual[key] == value }
    }
}

/**
 * Escapes user input for an FTS4 MATCH expression.
 *
 * Raw text goes straight into FTS query syntax, so an unescaped apostrophe or a
 * bare `NEAR`/`OR` turns a search into a syntax error or, worse, a different
 * query than the user typed. Each term is quoted and joined with OR.
 */
internal object FtsQuery {

    fun sanitize(raw: String): String? {
        val terms = raw
            .split(Regex("[^\\p{L}\\p{N}_]+"))
            .filter { it.isNotBlank() }
            .map { it.replace("\"", "") }
            .filter { it.isNotBlank() }

        if (terms.isEmpty()) return null
        return terms.joinToString(" OR ") { "\"$it\"" }
    }
}
