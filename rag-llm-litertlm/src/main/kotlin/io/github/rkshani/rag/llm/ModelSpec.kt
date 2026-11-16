package io.github.rkshani.rag.llm

import java.net.URI

/** Everything a download is checked against. All of it is required. */
data class ModelSpec(
    /** Must be https. A model is native code input; it is never fetched in clear text. */
    val url: String,
    /** Plain file name, no directories. Stored under `filesDir/models/`. */
    val fileName: String,
    /** Lowercase hex SHA-256 of the exact file. Downloads that do not match are deleted. */
    val sha256: String,
    /** Expected size in bytes, when known. Lets the download stop early on a wrong file. */
    val sizeBytes: Long? = null,
) {
    init {
        val uri = runCatching { URI(url) }.getOrNull()
        require(uri?.scheme.equals("https", ignoreCase = true) && !uri?.host.isNullOrBlank()) {
            "model URL must be https: $url"
        }
        require(FILE_NAME.matches(fileName) && !fileName.startsWith(".")) {
            "fileName must be a plain file name (letters, digits, . _ -): $fileName"
        }
        require(SHA256.matches(sha256)) { "sha256 must be 64 lowercase hex characters" }
        require(sizeBytes == null || sizeBytes > 0) { "sizeBytes must be positive" }
    }

    private companion object {
        val FILE_NAME = Regex("""[A-Za-z0-9._-]{1,128}""")
        val SHA256 = Regex("""[0-9a-f]{64}""")
    }
}
