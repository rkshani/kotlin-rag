package io.github.rkshani.rag.llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ModelSpecTest {

    private val hash = "a".repeat(64)

    @Test
    fun `accepts an https url with a plain file name`() {
        val spec = ModelSpec("https://example.com/models/gemma.litertlm", "gemma.litertlm", hash, 1_000)
        assertEquals("gemma.litertlm", spec.fileName)
    }

    @Test
    fun `rejects plain http`() {
        assertFailsWith<IllegalArgumentException> { ModelSpec("http://example.com/m.litertlm", "m.litertlm", hash) }
    }

    @Test
    fun `rejects other schemes and malformed urls`() {
        listOf("file:///sdcard/m.litertlm", "ftp://example.com/m", "https://", "not a url", "").forEach { url ->
            assertFailsWith<IllegalArgumentException>(url) { ModelSpec(url, "m.litertlm", hash) }
        }
    }

    @Test
    fun `rejects file names that could escape the models directory`() {
        listOf("../databases/kotlin-rag.db", "a/b.litertlm", "..", ".hidden", "", "x".repeat(200), "naïve.bin").forEach { name ->
            assertFailsWith<IllegalArgumentException>(name) { ModelSpec("https://example.com/m", name, hash) }
        }
    }

    @Test
    fun `requires a full lowercase sha256`() {
        listOf("", "abc", "A".repeat(64), "g".repeat(64), "a".repeat(63)).forEach { bad ->
            assertFailsWith<IllegalArgumentException>(bad) { ModelSpec("https://example.com/m", "m.bin", bad) }
        }
    }

    @Test
    fun `rejects a non-positive size`() {
        assertFailsWith<IllegalArgumentException> { ModelSpec("https://example.com/m", "m.bin", hash, 0) }
    }
}
