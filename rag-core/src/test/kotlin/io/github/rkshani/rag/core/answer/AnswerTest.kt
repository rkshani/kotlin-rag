package io.github.rkshani.rag.core.answer

import io.github.rkshani.rag.core.GenerationOptions
import io.github.rkshani.rag.core.Generator
import io.github.rkshani.rag.core.SearchFilter
import io.github.rkshani.rag.core.model.Chunk
import io.github.rkshani.rag.core.retrieval.RetrievalSource
import io.github.rkshani.rag.core.retrieval.RetrievedChunk
import io.github.rkshani.rag.core.retrieval.Retriever
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private fun retrieved(
    id: String,
    text: String,
    score: Float,
    docId: String = "doc1",
    charStart: Int = 0,
    page: Int? = null,
    metadata: Map<String, String> = emptyMap(),
) = RetrievedChunk(
    chunk = Chunk(
        id = id,
        docId = docId,
        index = 0,
        text = text,
        charStart = charStart,
        charEnd = charStart + text.length,
        page = page,
        metadata = metadata,
    ),
    score = score,
    ranks = mapOf(RetrievalSource.VECTOR to 1),
    sourceScores = mapOf(RetrievalSource.VECTOR to score),
)

class ContextAssemblerTest {

    private val assembler = ContextAssembler(order = ContextOrder.RELEVANCE, minPassageTokens = 1)

    @Test
    fun `trims text an earlier chunk already covers`() {
        // Chunk B starts inside chunk A, exactly as the chunker produces.
        val full = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron."
        val a = retrieved("c1", full.substring(0, 50), 0.9f, charStart = 0)
        val b = retrieved("c2", full.substring(30), 0.8f, charStart = 30)

        val context = assembler.assemble(listOf(a, b))

        assertEquals(2, context.passages.size)
        val joined = context.passages.sortedBy { it.chunk.charStart }.joinToString("") { it.text }
        assertEquals(
            full.filterNot { it.isWhitespace() },
            joined.filterNot { it.isWhitespace() },
            "trimmed passages should reconstruct the source without duplication",
        )
    }

    @Test
    fun `drops a chunk wholly contained in another`() {
        val outer = retrieved("outer", "a".repeat(200), 0.9f, charStart = 0)
        val inner = retrieved("inner", "a".repeat(50), 0.8f, charStart = 20)

        val context = assembler.assemble(listOf(outer, inner))

        assertEquals(listOf("outer"), context.passages.map { it.chunk.id })
        assertEquals(1, context.droppedCount)
    }

    @Test
    fun `trimming does not depend on the order results arrive in`() {
        val text = "one two three four five six seven eight nine ten eleven twelve thirteen"
        val a = retrieved("c1", text.substring(0, 40), 0.5f, charStart = 0)
        val b = retrieved("c2", text.substring(25), 0.9f, charStart = 25)

        val ranked = assembler.assemble(listOf(b, a)).passages.associate { it.chunk.id to it.text }
        val reversed = assembler.assemble(listOf(a, b)).passages.associate { it.chunk.id to it.text }

        assertEquals(ranked, reversed)
    }

    @Test
    fun `chunks from different documents are never trimmed against each other`() {
        // Identical offsets in different documents mean nothing in common.
        val a = retrieved("c1", "content of the first document here", 0.9f, docId = "doc1", charStart = 0)
        val b = retrieved("c2", "content of the second document here", 0.8f, docId = "doc2", charStart = 0)

        val context = assembler.assemble(listOf(a, b))

        assertEquals(2, context.passages.size)
        assertTrue(context.passages.all { it.text.isNotBlank() })
    }

    @Test
    fun `respects the token budget`() {
        val long = "word ".repeat(200)
        val assembler = ContextAssembler(tokenBudget = 100, order = ContextOrder.RELEVANCE)

        val context = assembler.assemble(
            listOf(
                retrieved("c1", long, 0.9f, docId = "d1"),
                retrieved("c2", long, 0.8f, docId = "d2"),
                retrieved("c3", long, 0.7f, docId = "d3"),
            ),
        )

        assertTrue(context.usedTokens <= 100, "used ${context.usedTokens} tokens against a budget of 100")
        assertTrue(context.droppedCount > 0)
    }

    @Test
    fun `keeps scanning past an oversized passage for a smaller one that fits`() {
        val assembler = ContextAssembler(tokenBudget = 60, order = ContextOrder.RELEVANCE, minPassageTokens = 1)

        val context = assembler.assemble(
            listOf(
                retrieved("huge", "word ".repeat(300), 0.9f, docId = "d1"),
                retrieved("small", "a short but useful passage about the topic at hand", 0.8f, docId = "d2"),
            ),
        )

        assertEquals(listOf("small"), context.passages.map { it.chunk.id })
    }

    @Test
    fun `edge ordering places the best first and second best last`() {
        val assembler = ContextAssembler(order = ContextOrder.EDGES, minPassageTokens = 1)
        val results = (1..5).map {
            retrieved("c$it", "passage number $it with enough text to survive the minimum", 1f - it * 0.1f, docId = "d$it")
        }

        val ids = assembler.assemble(results).passages.map { it.chunk.id }

        assertEquals(listOf("c1", "c3", "c5", "c4", "c2"), ids)
    }

    @Test
    fun `ordinals are assigned in final display order`() {
        val assembler = ContextAssembler(order = ContextOrder.EDGES, minPassageTokens = 1)
        val results = (1..3).map {
            retrieved("c$it", "passage number $it with enough text to survive", 1f - it * 0.1f, docId = "d$it")
        }

        val passages = assembler.assemble(results).passages

        assertEquals(listOf(1, 2, 3), passages.map { it.ordinal })
        assertEquals("c1", passages.first().chunk.id)
    }

    @Test
    fun `duplicate chunk ids are collapsed`() {
        val one = retrieved("same", "a passage with sufficient length to be kept", 0.9f)
        val context = assembler.assemble(listOf(one, one))
        assertEquals(1, context.passages.size)
    }

    @Test
    fun `empty input yields empty context`() {
        val context = assembler.assemble(emptyList())
        assertTrue(context.isEmpty)
        assertEquals(0, context.usedTokens)
    }

    @Test
    fun `passage label includes title and page when present`() {
        val context = assembler.assemble(
            listOf(
                retrieved(
                    "c1", "some passage text that is long enough to keep", 0.9f,
                    page = 7, metadata = mapOf("title" to "Lecture Notes"),
                ),
            ),
        )

        assertEquals("Lecture Notes, page 7", context.passages.single().label())
    }
}

class PromptBuilderTest {

    private fun context() = ContextAssembler(order = ContextOrder.RELEVANCE).assemble(
        listOf(
            retrieved("c1", "The mitochondrion generates most of the cell's ATP supply.", 0.9f, docId = "bio", page = 3),
            retrieved("c2", "Chloroplasts perform photosynthesis in plant cells only.", 0.8f, docId = "bio2", page = 5),
        ),
    )

    @Test
    fun `numbers passages and includes the question`() {
        val prompt = PromptBuilder(PromptTemplate.Plain).build("What makes ATP?", context())

        assertTrue(prompt.contains("[1]"))
        assertTrue(prompt.contains("[2]"))
        assertTrue(prompt.contains("What makes ATP?"))
        assertTrue(prompt.contains("page 3"))
    }

    @Test
    fun `instructs the model to stay grounded`() {
        val prompt = PromptBuilder(PromptTemplate.Plain).build("q", context())
        assertTrue(prompt.contains("only the context passages"))
    }

    @Test
    fun `gemma template folds the system text into the user turn`() {
        // Gemma has no system role; emitting one produces off-distribution input.
        val prompt = PromptBuilder(PromptTemplate.Gemma).build("q", context())

        assertTrue(prompt.startsWith("<start_of_turn>user"))
        assertTrue(prompt.trimEnd().endsWith("<start_of_turn>model"))
        assertFalse(prompt.contains("<start_of_turn>system"))
    }

    @Test
    fun `phi3 template uses its own markers`() {
        val prompt = PromptBuilder(PromptTemplate.Phi3).build("q", context())
        assertTrue(prompt.contains("<|system|>"))
        assertTrue(prompt.trimEnd().endsWith("<|assistant|>"))
    }
}

class CitationsTest {

    @Test
    fun `extracts markers in order without duplicates`() {
        val parsed = Citations.parse("As shown in [2], and also [1], and again [2].", passageCount = 3)
        assertEquals(listOf(2, 1), parsed.cited)
        assertTrue(parsed.invalid.isEmpty())
    }

    @Test
    fun `flags citations pointing at passages that were never supplied`() {
        // A fabricated source is the most damaging thing a RAG answer can do quietly.
        val parsed = Citations.parse("According to [7] and [1].", passageCount = 2)
        assertEquals(listOf(1), parsed.cited)
        assertEquals(listOf(7), parsed.invalid)
    }

    @Test
    fun `ignores bracketed text that is not a citation`() {
        val parsed = Citations.parse("An array like [x] or [] or [abc] is not a citation.", passageCount = 3)
        assertTrue(parsed.cited.isEmpty())
        assertTrue(parsed.invalid.isEmpty())
    }

    @Test
    fun `zero is not a valid ordinal`() {
        assertEquals(listOf(0), Citations.parse("see [0]", passageCount = 3).invalid)
    }

    @Test
    fun `uncited text yields no citations`() {
        assertTrue(Citations.parse("A plain answer with no markers.", passageCount = 3).cited.isEmpty())
    }
}

class RagAnswererTest {

    private class FakeRetriever(private val results: List<RetrievedChunk>) : Retriever {
        var lastK: Int? = null
        override suspend fun retrieve(query: String, k: Int, filter: SearchFilter?): List<RetrievedChunk> {
            lastK = k
            return results
        }
    }

    private class FakeGenerator(
        private val tokens: List<String>,
        private val available: Boolean = true,
        private val availabilityError: Throwable? = null,
    ) : Generator {
        var receivedPrompt: String? = null
        override suspend fun isAvailable(): Boolean {
            availabilityError?.let { throw it }
            return available
        }
        override fun generate(prompt: String, options: GenerationOptions): Flow<String> = flow {
            receivedPrompt = prompt
            tokens.forEach { emit(it) }
        }
        override fun close() = Unit
    }

    private val results = listOf(
        retrieved("c1", "Mitochondria produce ATP through oxidative phosphorylation.", 0.9f, docId = "d1"),
        retrieved("c2", "Chloroplasts convert light energy into chemical energy.", 0.7f, docId = "d2"),
    )

    @Test
    fun `emits context first then tokens then the completed answer`() = runTest {
        val answerer = RagAnswerer(
            retriever = FakeRetriever(results),
            generator = FakeGenerator(listOf("ATP comes ", "from mitochondria [1].")),
        )

        val events = answerer.answer("Where does ATP come from?").toList()

        assertTrue(events.first() is AnswerEvent.Context)
        assertEquals(2, events.count { it is AnswerEvent.Token })
        val completed = events.last() as AnswerEvent.Completed
        assertEquals(AnswerMode.GENERATED, completed.answer.mode)
        assertEquals("ATP comes from mitochondria [1].", completed.answer.text)
    }

    @Test
    fun `resolves citations to the passages that were supplied`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(results), FakeGenerator(listOf("See [1].")))

        val answer = (answerer.answer("q").toList().last() as AnswerEvent.Completed).answer

        assertEquals(1, answer.citations.size)
        assertEquals(1, answer.citations.single().ordinal)
        assertTrue(answer.invalidCitations.isEmpty())
    }

    @Test
    fun `reports fabricated citations`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(results), FakeGenerator(listOf("Per [9], yes.")))

        val answer = (answerer.answer("q").toList().last() as AnswerEvent.Completed).answer

        assertEquals(listOf(9), answer.invalidCitations)
    }

    @Test
    fun `falls back to extractive when no generator is configured`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(results), generator = null)

        val events = answerer.answer("q").toList()
        val answer = (events.last() as AnswerEvent.Completed).answer

        assertEquals(AnswerMode.EXTRACTIVE, answer.mode)
        assertEquals(0, events.count { it is AnswerEvent.Token })
        assertTrue(answer.text.contains("Mitochondria"))
    }

    @Test
    fun `falls back when the generator reports itself unavailable`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(results), FakeGenerator(listOf("unused"), available = false))

        val answer = (answerer.answer("q").toList().last() as AnswerEvent.Completed).answer

        assertEquals(AnswerMode.EXTRACTIVE, answer.mode)
    }

    @Test
    fun `falls back when the availability check itself throws`() = runTest {
        // A corrupt or partially downloaded model throws here rather than returning false.
        val answerer = RagAnswerer(
            FakeRetriever(results),
            FakeGenerator(listOf("unused"), availabilityError = IllegalStateException("bad model")),
        )

        val answer = (answerer.answer("q").toList().last() as AnswerEvent.Completed).answer

        assertEquals(AnswerMode.EXTRACTIVE, answer.mode)
    }

    @Test
    fun `falls back when generation produces nothing`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(results), FakeGenerator(listOf("", "   ")))

        val answer = (answerer.answer("q").toList().last() as AnswerEvent.Completed).answer

        assertEquals(AnswerMode.EXTRACTIVE, answer.mode)
    }

    @Test
    fun `says so when retrieval finds nothing`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(emptyList()), FakeGenerator(listOf("should not run")))

        val events = answerer.answer("q").toList()
        val answer = (events.last() as AnswerEvent.Completed).answer

        assertEquals(AnswerMode.NO_CONTEXT, answer.mode)
        assertEquals(0, events.count { it is AnswerEvent.Token })
    }

    @Test
    fun `extractive answers are ordered by relevance not prompt layout`() = runTest {
        val answerer = RagAnswerer(FakeRetriever(results), generator = null)

        val answer = (answerer.answer("q").toList().last() as AnswerEvent.Completed).answer

        assertEquals("c1", answer.citations.first().chunk.id)
    }

    @Test
    fun `the prompt carries the assembled passages`() = runTest {
        val generator = FakeGenerator(listOf("ok"))
        RagAnswerer(FakeRetriever(results), generator).answer("What is ATP?").toList()

        val prompt = generator.receivedPrompt.orEmpty()
        assertTrue(prompt.contains("Mitochondria"))
        assertTrue(prompt.contains("What is ATP?"))
    }
}
