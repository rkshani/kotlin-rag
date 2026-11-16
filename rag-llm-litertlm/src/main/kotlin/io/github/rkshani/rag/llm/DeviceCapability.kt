package io.github.rkshani.rag.llm

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.StatFs

/**
 * Whether this device can realistically run an on-device LLM.
 *
 * Worth checking before the download, not after: a 1.3 GB transfer that ends in
 * an out-of-memory crash is the worst possible outcome for a user on metered data.
 */
object DeviceCapability {

    data class Report(
        val totalRamBytes: Long,
        val availableStorageBytes: Long,
        val isLowRamDevice: Boolean,
        val is64Bit: Boolean,
    ) {
        fun canRun(modelBytes: Long, minimumRamBytes: Long = MIN_RAM_BYTES): Boolean = reason(modelBytes, minimumRamBytes) == null

        /** Why the model cannot run here, or null when it can. Show this to the user. */
        fun reason(modelBytes: Long, minimumRamBytes: Long = MIN_RAM_BYTES): String? = when {
            isLowRamDevice -> "device is flagged low-RAM"
            !is64Bit -> "64-bit ABI required"
            totalRamBytes < minimumRamBytes ->
                "needs ${minimumRamBytes / GB} GB RAM, device has ${totalRamBytes / GB} GB"
            else -> null
        }

        /** Storage check for the download, which needs room for the file plus slack. */
        fun canDownload(modelBytes: Long): Boolean = availableStorageBytes > modelBytes + modelBytes / 5
    }

    private const val MB = 1024L * 1024L
    private const val GB = MB * 1024L

    /** A 2B int4 model needs roughly 1.4 GB resident plus KV cache and the app itself. */
    const val MIN_RAM_BYTES: Long = 4L * GB

    fun inspect(context: Context): Report {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val stat = StatFs(context.filesDir.absolutePath)

        return Report(
            totalRamBytes = memoryInfo.totalMem,
            availableStorageBytes = stat.availableBlocksLong * stat.blockSizeLong,
            isLowRamDevice = activityManager.isLowRamDevice,
            is64Bit = Build.SUPPORTED_64_BIT_ABIS.isNotEmpty(),
        )
    }
}
