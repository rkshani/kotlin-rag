# kotlin-rag

On-device retrieval-augmented generation for Android, in Kotlin. Index PDFs, Word files, text, scanned pages and
photos of notes, search them with hybrid vector + keyword retrieval, and ask questions answered by a local LLM —
no API keys, no network at query time. Questions can be typed or spoken. Works in Urdu as well as
English.

```kotlin
val embedder = OnnxEmbedder.create(context, ModelSource.Asset("model_int8.onnx"), ModelSource.Asset("unigram_vocab.tsv"),
    EmbeddingModelSpec.MULTILINGUAL_E5_SMALL)

val rag = KotlinRag.builder()
    .embedder(embedder)
    .store(RoomRagStore.create(context, embedder.dimensions))   // vectors + BM25 + document list
    .tokenEstimator(embedder.tokenEstimator())                   // exact chunk sizes
    .addLoader(OcrPdf.loader(context, ocr))                      // PDFs; scanned pages go to OCR
    .addLoader(ImageOcrLoader(ocr))                              // photos; DOCX and TXT are built in
    .generator(VerifiedModelGenerator(context, modelSpec))       // optional; answers are extractive without it
    .build()

rag.index(documentInput).collect { progress -> /* Chunking / Embedding(done, total) / Completed */ }

val chat = rag.chat()
chat.ask("بیٹری کتنی دیر چلتی ہے؟").collect { event -> /* Context, Token, Completed */ }
chat.ask("and at night?").collect { /* follow-up resolved against the previous question */ }
```

## Modules

| Module | Contents |
|---|---|
| `rag-core` | Pure Kotlin. `KotlinRag` facade, chunker, indexer, loaders (TXT, DOCX), hybrid retrieval with RRF, re-ranker hook, chat + query rewriting, prompt-injection defences, in-memory store. |
| `rag-embed-onnx` | ONNX Runtime embedder and cross-encoder re-ranker. WordPiece (English BERT models) and SentencePiece Unigram (XLM-R models — Urdu) tokenizers in pure Kotlin. |
| `rag-store-room` | Room store: vector scan, FTS4 keyword index ranked with BM25, document catalog. |
| `rag-loader-pdf` | PDF text extraction with page numbers (PdfBox-Android), with a per-page fallback hook for scanned pages. |
| `rag-ocr` | OCR: ML Kit for English, Tesseract for Urdu and mixed pages. Scanned PDF pages and photos of notes. |
| `rag-voice` | Spoken questions, offline: Whisper on ONNX Runtime, or Android's on-device recognizer. |
| `rag-llm-litertlm` | Local LLM on LiteRT-LM, device capability check, verified resumable model download. |
| `demo` | Compose app: add documents, chat with citations, download the model. |

Requirements: JDK 17. Android modules: minSdk 26, `arm64-v8a`.

## Urdu

| Embedding model | Size (int8) | Tokenizer | Urdu |
|---|---|---|---|
| all-MiniLM-L6-v2 | ~23 MB | WordPiece | No — Arabic script becomes `[UNK]` |
| multilingual-e5-small | ~120 MB | SentencePiece Unigram | Yes |

`UnigramTokenizer` reads either the model's own `sentencepiece.bpe.model` or the
`unigram_vocab.tsv` the export script writes. It is checked id-for-id against the reference
`sentencepiece` library on Urdu, English, mixed script, digits, unknown characters and whitespace
edge cases. Its `nmt_nfkc` normalisation rules were measured by running the reference normaliser
over every Unicode code point.

Keyword search folds Urdu spelling variants before indexing and querying: zer/zabar/pesh and other
diacritics are dropped, Arabic-keyboard ي/ك/ه map to Urdu ی/ک/ہ, Urdu digits become ASCII, and PDF
presentation forms fold back to plain letters. So `کتاب` finds `کِتاب`.

## OCR — scanned PDFs and photos

```kotlin
val ocr = RoutingOcr(latin = MlKitOcr(), tesseract = TesseractOcr(TessData.install(context)))
builder.addLoader(OcrPdf.loader(context, ocr)).addLoader(ImageOcrLoader(ocr))

// Per document: which script the scan is in.
DocumentInput(..., metadata = mapOf(OcrLanguage.METADATA_KEY to OcrLanguage.URDU.name))
```

- **Per page, not per file.** A PDF page with real text is read directly; only pages with almost
  no text (scans) are rendered and OCR'd. A mostly-digital PDF with a few scanned pages works, and
  digital pages never pay the OCR cost.
- **Engine by language.** ML Kit (bundled, fast) reads English. It has no Arabic-script model, so
  Urdu and mixed Urdu + English pages go to Tesseract 5. Ship `urd.traineddata` (and
  `eng.traineddata` for mixed pages) in `assets/tessdata/`.
- **Rendering**: pages are drawn on white (PdfRenderer's transparent default reads as black to OCR),
  at ~300 dpi with the longest side capped so a page stays around 17 MB of bitmap.
- **Photos** are downsampled while decoding and turned upright from their EXIF orientation.
- **Clean-up**: hyphenated line breaks are rejoined (Latin only) and border/speck noise lines are
  dropped before indexing.
- **Progress and cancel**: OCR runs at seconds per page, so loaders report page progress
  (`IndexProgress.Loading`) and stop at the next page when indexing is cancelled.
- **Limit**: the stock Urdu model reads Naskh print well and Nastaliq (most Urdu books) poorly.

## Voice questions

```kotlin
val voice = VoiceInputs.best(listOf(WhisperVoiceInput(context, modelFile, promptFile), SystemVoiceInput(context)), VoiceLanguage.URDU)
voice?.listen(VoiceLanguage.URDU)?.collect { event -> /* Listening, Level, Partial, Transcribing, Final(text) */ }
```

- **Whisper on device** (`WhisperVoiceInput`): the all-in-one Whisper ONNX model from Microsoft
  Olive, run with ONNX Runtime + onnxruntime-extensions. Kotlin only feeds samples in and reads
  text out. `whisper-small` int8 (~250 MB) is the practical minimum for Urdu.
- **Language is forced.** Left to detect it, Whisper often hears Urdu as Hindi and writes
  Devanagari, which matches nothing in an Urdu index. The prompt ids come from the model's own
  tokenizer (`tools/prepare_whisper.py`); the built-in default (Urdu = 50290) matches tiny–medium.
- **Recording stops by itself** when the speaker pauses. The noise floor is measured over the first
  300 ms, so a fan is not mistaken for speech and a quiet room still triggers.
- **Silence hallucinations** ("Thank you.", subtitle credits) are filtered out of transcripts.
- **System recognizer** (`SystemVoiceInput`): strictly on-device (Android 12+). It never falls back
  to the online recognizer — that would send the user's voice to a server. On Android 13+ it can
  ask the phone to download the Urdu offline pack.

## Security and robustness

- **Prompt injection.** Chat-template control tokens are stripped from document text, so a file
  containing `<end_of_turn><start_of_turn>user` cannot forge a turn. Passages are fenced and the
  fence cannot be closed from inside. Passages that read like instructions to the model (English,
  Urdu, Roman Urdu) are withheld by default and reported on the answer. This is defence in depth,
  not a guarantee — models read instructions and data through the same channel.
- **Fabricated citations** (`[7]` when three passages were supplied) are reported, not hidden.
- **Hostile files.** Byte, character and page limits on every loader; DOCX zip bombs stop at an
  expanded-size limit; XML external entities are disabled (XXE).
- **Model download.** HTTPS only, SHA-256 required, file name validated against path traversal,
  size-capped, resumable. A verification marker is written only after the hash matches, and only a
  marked file is ever loaded into the engine.
- **Cancellation** reaches every long loop: embedding batches, the vector scan, the download, and
  native decoding (`cancelProcess`). A cancelled or failed index removes the partial document.
- **Graceful degradation.** A failing retrieval source, re-ranker or generator never fails the
  answer — it falls back and reports through `onFailure`.
- **Dispatchers** are injected (`RagDispatchers`) everywhere, so tests control threading.
- **ProGuard / R8**: consumer rules ship with the ONNX, LiteRT-LM and PDF modules.

## Design notes

**Offsets, not string search.** Every chunk stores `charStart`/`charEnd` into the original
document; citations and overlap trimming work on offsets.

**Chunks fit the embedder.** The default chunk size is derived from the embedder's window
(MiniLM: 256 tokens). Text past the window is silently dropped by the model, so a 400-token chunk
would lose a third of itself.

**Hybrid retrieval with RRF.** Vectors capture meaning but blur exact literals (course codes, names,
formulas); BM25 nails literals and misses paraphrase. Results are fused by rank, which needs no
score calibration. An optional cross-encoder re-ranks the fused head.

**No ANN index.** An exact scan of 10k × 384 floats takes a few milliseconds. The Room scan pages
through the table so memory stays flat. Worth revisiting above ~100k chunks.

**BM25 over FTS4.** FTS4 has no `bm25()`; ranking is computed from `matchinfo('pcnalx')`, whose
layout was confirmed against real SQLite.

**Context assembly** trims chunk overlap (≈48% of assembled text was duplicate) and places the best
passage first and the second-best last, where models attend most reliably.

**The LLM is optional.** Without a usable model, answers are the matching passages verbatim.

## Models

```bash
cd tools && pip install -r requirements.txt
python export_embedder.py --model intfloat/multilingual-e5-small          # Urdu + English
python export_embedder.py --model sentence-transformers/all-MiniLM-L6-v2  # English, smaller
python export_embedder.py --cross-encoder --model cross-encoder/ms-marco-MiniLM-L-6-v2
python prepare_whisper.py --olive-example-dir Olive/examples/whisper --model openai/whisper-small
```

Each run writes `model_int8.onnx`, the tokenizer file, and `parity.json` for the device test.

For the demo, copy the model and tokenizer file into `demo/src/main/assets/`. To enable generated
answers, set `demo.llmUrl`, `demo.llmFile`, `demo.llmSha256` and `demo.llmSize` (a `.litertlm`
model) in `~/.gradle/gradle.properties`.

## Tests

```bash
./gradlew :rag-core:test                          # JVM
./gradlew :rag-embed-onnx:test :rag-store-room:test :rag-llm-litertlm:test
./gradlew connectedAndroidTest                    # device: Room, embedding parity
```

What has actually been run so far, outside Gradle (see "Status"):

- `rag-core`: 115 tests — chunker invariants, loaders (encodings, DOCX, XXE, zip bomb, limits),
  Urdu normalisation, indexer (progress, cancel cleanup, re-index), the full `KotlinRag` flow,
  chat and follow-ups, prompt-injection cases, fusion, re-ranking, every fallback path.
- `rag-embed-onnx` tokenizers: 24 tests, including 20 fixtures matched against sentencepiece.
- `rag-store-room` BM25 / query escaping / top-K: 17 tests.
- `rag-llm-litertlm` download validation: 6 tests.
- `rag-ocr`: 12 tests — OCR text clean-up, render sizing, language selection, traineddata names.
- `rag-voice`: 16 tests — auto-stop (speech, pauses, noise, clicks, silence), PCM/WAV encoding,
  Whisper prompt ids, transcript clean-up.

## Status — read before relying on it

- No Gradle build has run yet: the build machine could not reach Maven or Google's repository.
  The pure-Kotlin code above was compiled and tested with `kotlinc` directly. The Android modules
  were type-checked against stubs shaped like the real APIs, which catches most mistakes but not
  all. Expect a few fixes on the first real build.
- `rag-llm-litertlm` follows the LiteRT-LM Kotlin API as currently documented and pins `0.19.0`.
  If that release differs (for example `temperature` as `Float` instead of `Double`), only
  `LiteRtLmGenerator.kt` needs to change.
- The Room and embedding-parity device tests have not run on a phone yet. OCR and voice have only
  been checked off-device (their logic tested, their Android code type-checked).
- Versions to confirm on the first build: Tesseract4Android `4.9.0` from Maven Central (older
  releases were on JitPack), onnxruntime-extensions-android `0.14.0`.
- The Whisper model has to be built with Olive; its exact input names depend on the Olive version.
  `prepare_whisper.py` prints them, and `WhisperOnnx` only sends inputs the model declares.
- Not done yet: benchmarks, Maven Central publishing, Dokka, `explicitApi()`, OCR for scanned PDFs.

## License

Apache 2.0
