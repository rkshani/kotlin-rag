package io.github.rkshani.rag.demo

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.rkshani.rag.core.DocumentInfo
import io.github.rkshani.rag.core.answer.AnswerEvent
import io.github.rkshani.rag.core.answer.AnswerMode
import io.github.rkshani.rag.core.index.IndexProgress
import io.github.rkshani.rag.core.loader.DocumentInput
import io.github.rkshani.rag.core.retrieval.RetrievalSource
import io.github.rkshani.rag.llm.DeviceCapability
import io.github.rkshani.rag.llm.ModelDownloader
import io.github.rkshani.rag.ocr.OcrLanguage
import io.github.rkshani.rag.voice.SystemVoiceInput
import io.github.rkshani.rag.voice.VoiceEvent
import io.github.rkshani.rag.voice.VoiceInput
import io.github.rkshani.rag.voice.VoiceInputs
import io.github.rkshani.rag.voice.VoiceLanguage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class Citation(val ordinal: Int, val label: String, val snippet: String, val provenance: String)

data class ChatMessage(
    val id: Long,
    val question: String,
    val answer: String = "",
    val streaming: Boolean = true,
    val mode: AnswerMode? = null,
    val citations: List<Citation> = emptyList(),
    val withheld: Int = 0,
    val invalidCitations: List<Int> = emptyList(),
    val interrupted: Boolean = false,
    val retrievalQuery: String = "",
)

data class IndexingState(val name: String, val fraction: Float?, val error: String? = null, val step: String = "Indexing")

data class VoiceState(
    val listening: Boolean = false,
    val transcribing: Boolean = false,
    val level: Float = 0f,
    val partial: String = "",
    val error: String? = null,
    val engine: String? = null,
    /** True when Urdu speech needs the phone's offline pack downloaded first. */
    val canDownloadOfflinePack: Boolean = false,
)

data class ModelState(
    val configured: Boolean,
    val downloaded: Boolean,
    val progress: Int? = null,
    val error: String? = null,
    val deviceNote: String? = null,
)

class RagViewModel(application: Application) : AndroidViewModel(application) {

    private val graph = (application as DemoApp).graph
    private val rag = graph.rag
    private var chat = rag?.chat()

    val setupError: String? = graph.setupError

    private val _documents = MutableStateFlow<List<DocumentInfo>>(emptyList())
    val documents: StateFlow<List<DocumentInfo>> = _documents

    private val _indexing = MutableStateFlow<IndexingState?>(null)
    val indexing: StateFlow<IndexingState?> = _indexing

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _model = MutableStateFlow(ModelState(configured = graph.llm != null, downloaded = false))
    val model: StateFlow<ModelState> = _model

    private val _documentLanguage = MutableStateFlow(OcrLanguage.ENGLISH)
    /** How scanned pages and photos are read. Chosen per document before adding it. */
    val documentLanguage: StateFlow<OcrLanguage> = _documentLanguage

    private val _voiceLanguage = MutableStateFlow(VoiceLanguage.URDU)
    val voiceLanguage: StateFlow<VoiceLanguage> = _voiceLanguage

    private val _voice = MutableStateFlow(VoiceState())
    val voice: StateFlow<VoiceState> = _voice

    private var indexJob: Job? = null
    private var askJob: Job? = null
    private var voiceJob: Job? = null
    private var activeVoice: VoiceInput? = null

    init {
        refreshDocuments()
        refreshModel()
        graph.llm?.let {
            viewModelScope.launch {
                ModelDownloader.observe(application).collect { infos ->
                    val info = infos.lastOrNull() ?: return@collect
                    _model.update { state ->
                        state.copy(
                            progress = if (info.state.isFinished) null else ModelDownloader.progressOf(info),
                            error = ModelDownloader.errorOf(info),
                        )
                    }
                    if (info.state.isFinished) refreshModel()
                }
            }
        }
    }

    fun refreshDocuments() {
        val rag = rag ?: return
        viewModelScope.launch { _documents.value = rag.documents() }
    }

    fun addDocument(uri: Uri) {
        val rag = rag ?: return
        val input = documentInput(uri)
        if (!rag.canLoad(input)) {
            _indexing.value = IndexingState(input.name, null, "This file type is not supported (PDF, DOCX, TXT, MD, photos).")
            return
        }
        indexJob?.cancel()
        indexJob = viewModelScope.launch {
            _indexing.value = IndexingState(input.name, null)
            try {
                rag.index(input).collect { progress ->
                    _indexing.value = when (progress) {
                        is IndexProgress.Loading -> IndexingState(input.name, progress.fraction, step = "Reading page ${progress.done} of ${progress.total}")
                        is IndexProgress.Chunking -> IndexingState(input.name, null, step = "Splitting")
                        is IndexProgress.Embedding -> IndexingState(input.name, progress.fraction, step = "Indexing")
                        is IndexProgress.Completed -> null
                    }
                }
            } catch (cancelled: CancellationException) {
                _indexing.value = null
                throw cancelled
            } catch (failure: Throwable) {
                _indexing.value = IndexingState(input.name, null, failure.message ?: "Indexing failed")
            }
            refreshDocuments()
        }
    }

    fun setDocumentLanguage(language: OcrLanguage) {
        _documentLanguage.value = language
    }

    fun setVoiceLanguage(language: VoiceLanguage) {
        _voiceLanguage.value = language
    }

    /** Listens, then asks the recognised question. Called after the mic permission is granted. */
    fun startVoice() {
        if (voiceJob?.isActive == true) return
        voiceJob = viewModelScope.launch {
            val language = _voiceLanguage.value
            val input = VoiceInputs.best(graph.voiceInputs, language)
            if (input == null) {
                val system = graph.voiceInputs.filterIsInstance<SystemVoiceInput>().firstOrNull()
                _voice.value = VoiceState(
                    error = "No offline speech recognizer for ${language.name.lowercase()} on this phone. " +
                        "Install the Whisper model, or download the phone's offline speech pack.",
                    canDownloadOfflinePack = system != null && android.os.Build.VERSION.SDK_INT >= 33,
                )
                return@launch
            }
            activeVoice = input
            _voice.value = VoiceState(listening = true, engine = input.name)
            try {
                input.listen(language).collect { event ->
                    when (event) {
                        VoiceEvent.Listening -> _voice.update { it.copy(listening = true) }
                        is VoiceEvent.Level -> _voice.update { it.copy(level = event.value) }
                        is VoiceEvent.Partial -> _voice.update { it.copy(partial = event.text) }
                        VoiceEvent.Transcribing -> _voice.update { it.copy(listening = false, transcribing = true, level = 0f) }
                        is VoiceEvent.Final -> {
                            _voice.value = if (event.text.isBlank()) VoiceState(error = "Didn't catch that — try again.") else VoiceState()
                            if (event.text.isNotBlank()) ask(event.text)
                        }
                        is VoiceEvent.Failed -> { _voice.value = VoiceState(error = event.message, engine = input.name) }
                    }
                }
            } finally {
                activeVoice = null
                _voice.update { if (it.listening || it.transcribing) VoiceState() else it }
            }
        }
    }

    /** Stop listening now and transcribe what was said so far. */
    fun finishVoice() {
        activeVoice?.finish()
    }

    fun cancelVoice() {
        voiceJob?.cancel()
        _voice.value = VoiceState()
    }

    fun voicePermissionDenied() {
        _voice.value = VoiceState(error = "Microphone permission is needed to ask by voice.")
    }

    fun downloadOfflineSpeechPack() {
        graph.voiceInputs.filterIsInstance<SystemVoiceInput>().firstOrNull()?.requestOfflineModel(_voiceLanguage.value)
        _voice.value = VoiceState(error = "Download started in system settings. Try again when it finishes.")
    }

    fun cancelIndexing() {
        indexJob?.cancel()
        _indexing.value = null
        refreshDocuments()
    }

    fun dismissIndexingError() {
        _indexing.value = null
    }

    fun remove(docId: String) {
        val rag = rag ?: return
        viewModelScope.launch {
            rag.remove(docId)
            refreshDocuments()
        }
    }

    fun ask(question: String) {
        val session = chat ?: return
        if (question.isBlank() || askJob?.isActive == true) return
        val id = System.nanoTime()
        _messages.update { it + ChatMessage(id, question.trim()) }

        askJob = viewModelScope.launch {
            try {
                session.ask(question).collect { event -> onEvent(id, event) }
            } catch (cancelled: CancellationException) {
                edit(id) { it.copy(streaming = false, interrupted = true) }
                throw cancelled
            } catch (failure: Throwable) {
                edit(id) { it.copy(streaming = false, answer = "Error: ${failure.message}") }
            }
        }
    }

    fun stop() {
        askJob?.cancel()
    }

    fun newChat() {
        stop()
        chat = rag?.chat()
        _messages.value = emptyList()
    }

    fun downloadModel() {
        val spec = graph.llm ?: return
        val report = DeviceCapability.inspect(getApplication<Application>())
        val size = spec.sizeBytes ?: 0L
        if (size > 0 && !report.canDownload(size)) {
            _model.update { it.copy(error = "Not enough free storage for the model.") }
            return
        }
        report.reason(size)?.let { reason -> _model.update { it.copy(deviceNote = "This device may not run it: $reason") } }
        ModelDownloader.enqueue(getApplication<Application>(), spec)
    }

    fun cancelDownload() = ModelDownloader.cancel(getApplication<Application>())

    fun deleteModel() {
        graph.llm?.let { ModelDownloader.delete(getApplication<Application>(), it) }
        refreshModel()
    }

    private fun refreshModel() {
        val spec = graph.llm ?: return
        val downloaded = ModelDownloader.verifiedFile(getApplication<Application>(), spec) != null
        val note = if (downloaded) {
            DeviceCapability.inspect(getApplication<Application>()).reason(spec.sizeBytes ?: 0L)?.let { "Downloaded, but this device can't run it: $it" }
        } else {
            null
        }
        _model.update { it.copy(downloaded = downloaded, deviceNote = note ?: graph.unavailableReason) }
    }

    private fun onEvent(id: Long, event: AnswerEvent) = when (event) {
        is AnswerEvent.Context -> edit(id) {
            it.copy(withheld = event.context.withheld.size, retrievalQuery = event.retrievalQuery)
        }
        is AnswerEvent.Token -> edit(id) { it.copy(answer = it.answer + event.text) }
        is AnswerEvent.Completed -> edit(id) {
            val answer = event.answer
            it.copy(
                answer = answer.text,
                streaming = false,
                mode = answer.mode,
                interrupted = answer.interrupted,
                invalidCitations = answer.invalidCitations,
                citations = answer.citations.map { passage ->
                    Citation(
                        ordinal = passage.ordinal,
                        label = passage.label(),
                        snippet = passage.text.trim().take(220),
                        provenance = buildList {
                            passage.source.ranks[RetrievalSource.VECTOR]?.let { rank -> add("vector #$rank") }
                            passage.source.ranks[RetrievalSource.KEYWORD]?.let { rank -> add("keyword #$rank") }
                            passage.source.rerankScore?.let { score -> add("rerank %.2f".format(score)) }
                        }.joinToString(" · "),
                    )
                },
            )
        }
    }

    private fun edit(id: Long, change: (ChatMessage) -> ChatMessage) {
        _messages.update { list -> list.map { if (it.id == id) change(it) else it } }
    }

    private fun documentInput(uri: Uri): DocumentInput {
        val resolver = getApplication<Application>().contentResolver
        var name = uri.lastPathSegment ?: "document"
        var size: Long? = null
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                cursor.getString(0)?.let { name = it }
                if (!cursor.isNull(1)) size = cursor.getLong(1)
            }
        }
        return DocumentInput(
            // The content Uri is a stable id for "this file": re-adding it replaces the old index.
            id = uri.toString(),
            name = name,
            sizeBytes = size,
            mimeType = resolver.getType(uri),
            metadata = mapOf(OcrLanguage.METADATA_KEY to _documentLanguage.value.name),
        ) { requireNotNull(resolver.openInputStream(uri)) { "cannot open $name" } }
    }
}
