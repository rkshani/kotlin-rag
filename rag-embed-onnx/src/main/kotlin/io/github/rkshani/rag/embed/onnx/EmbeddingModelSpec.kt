package io.github.rkshani.rag.embed.onnx

/**
 * Where a model or vocab file lives.
 *
 * Prefer [LocalFile] for the ONNX model: assets are compressed into the APK and
 * ORT needs the bytes in memory anyway, so a 90 MB asset costs you 90 MB of APK
 * plus a 90 MB heap spike. Download to filesDir instead.
 */
sealed interface ModelSource {
    data class Asset(val name: String) : ModelSource
    data class LocalFile(val path: String) : ModelSource
}

enum class Pooling { MEAN, CLS }

/** Which tokenizer file accompanies the model, and so which tokenizer reads it. */
enum class TokenizerFormat {
    /** BERT-style `vocab.txt` -> [WordPieceTokenizer]. */
    WORDPIECE_VOCAB,

    /** sentencepiece `.model` (XLM-R's `sentencepiece.bpe.model`) -> [UnigramTokenizer], fairseq ids. */
    SENTENCEPIECE_MODEL,

    /** `unigram_vocab.tsv` written by `tools/export_embedder.py` -> [UnigramTokenizer]. */
    UNIGRAM_TSV,
}

/**
 * Everything that must match between the exported ONNX file and the runtime.
 * Getting [dimensions] or [pooling] wrong produces vectors that look fine and
 * retrieve badly, so treat these as part of the model artifact.
 */
data class EmbeddingModelSpec(
    val dimensions: Int,
    val maxSequenceTokens: Int,
    val pooling: Pooling = Pooling.MEAN,
    val doLowerCase: Boolean = true,
    /** Asymmetric models need these; symmetric ones leave them empty. */
    val queryPrefix: String = "",
    val documentPrefix: String = "",
    val batchSize: Int = 8,
    val tokenizer: TokenizerFormat = TokenizerFormat.WORDPIECE_VOCAB,
) {
    companion object {
        /**
         * sentence-transformers/all-MiniLM-L6-v2 — 6 layers, 384 dims, ~23 MB int8.
         *
         * WARNING: English-only. Its vocab is bert-base-uncased (30522 WordPiece
         * tokens) with essentially no Urdu/Arabic-script coverage, so Urdu input
         * collapses to [UNK] and retrieval quality falls off a cliff. Use it for
         * English corpora only.
         */
        val ALL_MINILM_L6_V2 = EmbeddingModelSpec(
            dimensions = 384,
            maxSequenceTokens = 256,
        )

        /**
         * intfloat/multilingual-e5-small — 12 layers, 384 dims, XLM-R vocab
         * (250k SentencePiece Unigram pieces), ~120 MB int8. Covers Urdu properly.
         *
         * Uses [UnigramTokenizer]; ship the `unigram_vocab.tsv` the export script
         * writes, or switch [tokenizer] to SENTENCEPIECE_MODEL and ship the
         * model's own `sentencepiece.bpe.model`.
         */
        val MULTILINGUAL_E5_SMALL = EmbeddingModelSpec(
            dimensions = 384,
            maxSequenceTokens = 512,
            queryPrefix = "query: ",
            documentPrefix = "passage: ",
            batchSize = 4,
            tokenizer = TokenizerFormat.UNIGRAM_TSV,
        )
    }
}
