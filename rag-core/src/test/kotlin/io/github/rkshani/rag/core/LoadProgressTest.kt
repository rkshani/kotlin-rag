package io.github.rkshani.rag.core

import io.github.rkshani.rag.core.index.IndexProgress
import io.github.rkshani.rag.core.loader.DocumentInput
import io.github.rkshani.rag.core.loader.DocumentLoader
import io.github.rkshani.rag.core.loader.LoadCancelledException
import io.github.rkshani.rag.core.loader.LoadLimits
import io.github.rkshani.rag.core.loader.LoadProgress
import io.github.rkshani.rag.core.model.SourceDocument
import io.github.rkshani.rag.core.store.InMemoryStore
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import java.io.ByteArrayInputStream
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Stands in for an OCR loader: slow pages, progress per page, honours interruption. */
private class SlowPagedLoader(private val pages: Int, private val pageMillis: Long) : DocumentLoader {
    val pagesRead = AtomicInteger()

    override fun supports(input: DocumentInput) = input.extension == "scan"

    override fun load(input: DocumentInput, limits: LoadLimits) = load(input, limits, LoadProgress.NONE)

    override fun load(input: DocumentInput, limits: LoadLimits, progress: LoadProgress): SourceDocument {
        val text = StringBuilder()
        for (page in 1..pages) {
            try {
                Thread.sleep(pageMillis)
            } catch (interrupted: InterruptedException) {
                throw LoadCancelledException("stopped at page $page")
            }
            text.append("Page $page talks about solar batteries and wiring.\n\n")
            pagesRead.incrementAndGet()
            progress.onProgress(page, pages)
        }
        return SourceDocument(input.id, text.toString(), title = input.title)
    }
}

class LoadProgressTest {

    private fun input() = DocumentInput("scan-1", "notes.scan") { ByteArrayInputStream(ByteArray(0)) }

    @Test
    fun `page progress from the loader is reported before indexing`() = runBlocking {
        val loader = SlowPagedLoader(pages = 4, pageMillis = 1)
        val rag = KotlinRag.builder().embedder(HashingEmbedder()).store(InMemoryStore()).addLoader(loader).build()

        val events = rag.index(input()).toList()

        val loading = events.filterIsInstance<IndexProgress.Loading>()
        assertEquals(listOf(1, 2, 3, 4), loading.map { it.done })
        assertEquals(4, loading.last().total)
        assertTrue(events.last() is IndexProgress.Completed)
        assertTrue(rag.search("solar").isNotEmpty())
    }

    @Test
    fun `cancelling stops a slow loader instead of letting it finish in the background`() = runBlocking {
        val loader = SlowPagedLoader(pages = 200, pageMillis = 20)
        val rag = KotlinRag.builder().embedder(HashingEmbedder()).store(InMemoryStore()).addLoader(loader).build()

        rag.index(input()).take(2).collect()
        val atCancel = loader.pagesRead.get()
        Thread.sleep(300)

        assertTrue(loader.pagesRead.get() <= atCancel + 1, "loader kept running after cancel: ${loader.pagesRead.get()} pages")
        assertTrue(loader.pagesRead.get() < 200)
        assertEquals(0, rag.documents().size)
    }
}
