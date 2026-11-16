package io.github.rkshani.rag.llm

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.security.MessageDigest

/**
 * Downloads a model file to app storage, resumably and verified.
 *
 * A gigabyte-plus over a mobile connection will be interrupted — that is the
 * normal case. So the download runs in WorkManager (survives process death),
 * appends to a partial file with a Range request (survives interruption), and
 * checks SHA-256 before the file is ever used. A file that fails the check is
 * deleted: a truncated or tampered model otherwise surfaces later as a native
 * crash inside the inference engine, or worse, as a model that runs.
 *
 * Only after verification is a `.sha256` marker written next to the file;
 * [ModelDownloader.verifiedFile] requires it, so a file that arrived any other
 * way is never handed to the engine.
 */
class ModelDownloadWorker(
    context: Context,
    params: WorkerParameters,
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val spec = try {
            ModelSpec(
                url = inputData.getString(KEY_URL).orEmpty(),
                fileName = inputData.getString(KEY_FILE_NAME).orEmpty(),
                sha256 = inputData.getString(KEY_SHA256).orEmpty(),
                sizeBytes = inputData.getLong(KEY_SIZE, -1L).takeIf { it > 0 },
            )
        } catch (invalid: IllegalArgumentException) {
            return@withContext Result.failure(workDataOf(KEY_ERROR to invalid.message))
        }

        val directory = ModelDownloader.modelsDir(applicationContext)
        if (ModelDownloader.verifiedFile(applicationContext, spec) != null) {
            return@withContext Result.success(workDataOf(KEY_PATH to File(directory, spec.fileName).absolutePath))
        }
        val target = File(directory, spec.fileName)
        val partial = File(directory, "${spec.fileName}.part")
        ModelDownloader.marker(directory, spec.fileName).delete()

        try {
            download(spec, partial)
        } catch (tooLarge: ModelSizeException) {
            partial.delete()
            return@withContext Result.failure(workDataOf(KEY_ERROR to tooLarge.message))
        } catch (io: IOException) {
            // Keep the partial file: the retry resumes from where this stopped.
            return@withContext if (runAttemptCount < MAX_ATTEMPTS) Result.retry() else Result.failure(workDataOf(KEY_ERROR to io.message))
        }

        val actual = sha256(partial)
        if (actual != spec.sha256) {
            // A corrupt partial can never be repaired by resuming.
            partial.delete()
            return@withContext Result.failure(workDataOf(KEY_ERROR to "checksum mismatch: expected ${spec.sha256}, got $actual"))
        }

        target.delete()
        if (!partial.renameTo(target)) {
            return@withContext Result.failure(workDataOf(KEY_ERROR to "could not finalise download"))
        }
        ModelDownloader.marker(directory, spec.fileName).writeText("${spec.sha256} ${target.length()}")
        Result.success(workDataOf(KEY_PATH to target.absolutePath))
    }

    private suspend fun download(spec: ModelSpec, partial: File) {
        val existing = if (partial.exists()) partial.length() else 0L
        val connection = (URI(spec.url).toURL().openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            if (existing > 0) setRequestProperty("Range", "bytes=$existing-")
        }

        try {
            connection.connect()
            // HttpURLConnection never follows https -> http, but a redirect to
            // another https host is followed; the checksum covers that case.
            if (!connection.url.protocol.equals("https", ignoreCase = true)) throw IOException("redirected off https")
            val code = connection.responseCode
            if (code !in 200..299) throw IOException("HTTP $code")

            // Resume only if the server resumed from exactly where we stopped.
            val resumed = code == HttpURLConnection.HTTP_PARTIAL &&
                connection.getHeaderField("Content-Range")?.startsWith("bytes $existing-") == true
            if (!resumed && partial.exists()) partial.delete()

            val total = connection.contentLengthLong.takeIf { it > 0 }?.plus(if (resumed) existing else 0L)
            val limit = spec.sizeBytes ?: MAX_MODEL_BYTES
            if (total != null && total > limit) throw ModelSizeException("server reports $total bytes, expected at most $limit")
            var written = if (resumed) existing else 0L

            connection.inputStream.use { input ->
                FileOutputStream(partial, resumed).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    var lastReported = -1
                    while (true) {
                        // Cancelling the work (or WorkManager stopping it) ends the copy here.
                        currentCoroutineContext().ensureActive()
                        val read = input.read(buffer)
                        if (read < 0) break
                        written += read
                        if (written > limit) throw ModelSizeException("download exceeded $limit bytes")
                        output.write(buffer, 0, read)

                        val expected = total ?: spec.sizeBytes
                        if (expected != null && expected > 0) {
                            val percent = ((written * 100) / expected).toInt().coerceIn(0, 100)
                            if (percent != lastReported) {
                                lastReported = percent
                                setProgress(workDataOf(KEY_PROGRESS to percent, KEY_BYTES to written))
                            }
                        }
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private class ModelSizeException(message: String) : IOException(message)

    companion object {
        const val KEY_URL = "url"
        const val KEY_FILE_NAME = "fileName"
        const val KEY_SHA256 = "sha256"
        const val KEY_SIZE = "size"
        const val KEY_PATH = "path"
        const val KEY_PROGRESS = "progress"
        const val KEY_BYTES = "bytes"
        const val KEY_ERROR = "error"

        const val WORK_NAME = "kotlin-rag-model-download"
        private const val MAX_ATTEMPTS = 8

        /** Upper bound when the spec gives no size: no on-device LLM here is near this. */
        private const val MAX_MODEL_BYTES = 6L * 1024 * 1024 * 1024

        internal fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(256 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}

object ModelDownloader {

    /**
     * @param requiresUnmeteredNetwork defaults true. Silently pulling a gigabyte
     *   over cellular is the kind of thing that gets an app uninstalled.
     */
    fun enqueue(context: Context, spec: ModelSpec, requiresUnmeteredNetwork: Boolean = true) {
        val request = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(
                Data.Builder()
                    .putString(ModelDownloadWorker.KEY_URL, spec.url)
                    .putString(ModelDownloadWorker.KEY_FILE_NAME, spec.fileName)
                    .putString(ModelDownloadWorker.KEY_SHA256, spec.sha256)
                    .putLong(ModelDownloadWorker.KEY_SIZE, spec.sizeBytes ?: -1L)
                    .build(),
            )
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(if (requiresUnmeteredNetwork) NetworkType.UNMETERED else NetworkType.CONNECTED)
                    .setRequiresStorageNotLow(true)
                    .build(),
            )
            .build()

        WorkManager.getInstance(context).enqueueUniqueWork(
            ModelDownloadWorker.WORK_NAME,
            // KEEP, not REPLACE: re-entering the screen should not restart a
            // download that is already 800 MB in.
            ExistingWorkPolicy.KEEP,
            request,
        )
    }

    fun cancel(context: Context) {
        WorkManager.getInstance(context).cancelUniqueWork(ModelDownloadWorker.WORK_NAME)
    }

    fun observe(context: Context) =
        WorkManager.getInstance(context).getWorkInfosForUniqueWorkFlow(ModelDownloadWorker.WORK_NAME)

    fun progressOf(info: WorkInfo): Int = when (info.state) {
        WorkInfo.State.SUCCEEDED -> 100
        WorkInfo.State.RUNNING -> info.progress.getInt(ModelDownloadWorker.KEY_PROGRESS, 0)
        else -> 0
    }

    fun errorOf(info: WorkInfo): String? = info.outputData.getString(ModelDownloadWorker.KEY_ERROR)

    /**
     * The model file, only if this downloader verified it and it has not changed
     * size since. Pass the result to [LiteRtLmGenerator]; never a path from elsewhere.
     */
    fun verifiedFile(context: Context, spec: ModelSpec): File? {
        val directory = modelsDir(context)
        val file = File(directory, spec.fileName)
        val marker = marker(directory, spec.fileName)
        if (!file.isFile || !marker.isFile) return null
        val (hash, size) = marker.readText().trim().split(' ').let { it.getOrNull(0) to it.getOrNull(1)?.toLongOrNull() }
        return file.takeIf { hash == spec.sha256 && size == it.length() }
    }

    /** Re-hashes the file. Slow on a gigabyte — for a settings "verify model" action, not every launch. */
    suspend fun reverify(context: Context, spec: ModelSpec): Boolean = withContext(Dispatchers.IO) {
        val file = verifiedFile(context, spec) ?: return@withContext false
        val ok = ModelDownloadWorker.sha256(file) == spec.sha256
        if (!ok) delete(context, spec)
        ok
    }

    fun delete(context: Context, spec: ModelSpec) {
        val directory = modelsDir(context)
        File(directory, spec.fileName).delete()
        File(directory, "${spec.fileName}.part").delete()
        marker(directory, spec.fileName).delete()
    }

    internal fun modelsDir(context: Context): File = File(context.filesDir, "models").apply { mkdirs() }

    internal fun marker(directory: File, fileName: String) = File(directory, "$fileName.sha256")
}
