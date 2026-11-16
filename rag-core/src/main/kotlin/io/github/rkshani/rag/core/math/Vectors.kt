package io.github.rkshani.rag.core.math

import kotlin.math.sqrt

object Vectors {

    fun dot(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "dimension mismatch: ${a.size} vs ${b.size}" }
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }

    fun norm(a: FloatArray): Float {
        var sum = 0f
        for (v in a) sum += v * v
        return sqrt(sum)
    }

    /** In-place L2 normalisation. After this, [dot] equals cosine similarity. */
    fun normalize(a: FloatArray): FloatArray {
        val n = norm(a)
        if (n > 1e-8f) for (i in a.indices) a[i] = a[i] / n
        return a
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        val denom = norm(a) * norm(b)
        return if (denom < 1e-8f) 0f else dot(a, b) / denom
    }

    /** Little-endian float serialisation for BLOB storage in Room/SQLite. */
    fun toBytes(a: FloatArray): ByteArray {
        val out = ByteArray(a.size * 4)
        for (i in a.indices) {
            val bits = a[i].toRawBits()
            val o = i * 4
            out[o] = (bits and 0xFF).toByte()
            out[o + 1] = ((bits ushr 8) and 0xFF).toByte()
            out[o + 2] = ((bits ushr 16) and 0xFF).toByte()
            out[o + 3] = ((bits ushr 24) and 0xFF).toByte()
        }
        return out
    }

    fun fromBytes(bytes: ByteArray): FloatArray {
        require(bytes.size % 4 == 0) { "byte length must be a multiple of 4" }
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            val o = i * 4
            val bits = (bytes[o].toInt() and 0xFF) or
                ((bytes[o + 1].toInt() and 0xFF) shl 8) or
                ((bytes[o + 2].toInt() and 0xFF) shl 16) or
                ((bytes[o + 3].toInt() and 0xFF) shl 24)
            out[i] = Float.fromBits(bits)
        }
        return out
    }
}
