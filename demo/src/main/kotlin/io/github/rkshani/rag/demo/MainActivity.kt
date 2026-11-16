package io.github.rkshani.rag.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import android.Manifest
import android.content.pm.PackageManager
import io.github.rkshani.rag.ocr.OcrLanguage
import io.github.rkshani.rag.voice.VoiceLanguage
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.rkshani.rag.core.DocumentInfo
import io.github.rkshani.rag.core.answer.AnswerMode

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                DemoScreen()
            }
        }
    }
}

private enum class Tab(val label: String) { DOCUMENTS("Documents"), ASK("Ask"), MODEL("Model") }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DemoScreen(viewModel: RagViewModel = viewModel()) {
    var tab by rememberSaveable { mutableStateOf(Tab.DOCUMENTS) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("kotlin-rag · ${tab.label}") }) },
        bottomBar = {
            NavigationBar {
                Tab.entries.forEach { item ->
                    NavigationBarItem(
                        selected = tab == item,
                        onClick = { tab = item },
                        icon = {
                            Icon(
                                when (item) {
                                    Tab.DOCUMENTS -> Icons.Default.Description
                                    Tab.ASK -> Icons.Default.Chat
                                    Tab.MODEL -> Icons.Default.Memory
                                },
                                contentDescription = null,
                            )
                        },
                        label = { Text(item.label) },
                    )
                }
            }
        },
    ) { padding ->
        val modifier = Modifier.padding(padding).fillMaxSize()
        val setupError = viewModel.setupError
        when {
            setupError != null -> SetupHelp(setupError, modifier)
            tab == Tab.DOCUMENTS -> DocumentsTab(viewModel, modifier)
            tab == Tab.ASK -> AskTab(viewModel, modifier)
            else -> ModelTab(viewModel, modifier)
        }
    }
}

@Composable
private fun SetupHelp(error: String, modifier: Modifier) {
    Column(modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("The embedding model isn't set up", style = MaterialTheme.typography.titleMedium)
        Text(error, color = MaterialTheme.colorScheme.error)
        Text(
            "Run tools/export_embedder.py, then copy model_int8.onnx and vocab.txt " +
                "(or unigram_vocab.tsv for Urdu) into demo/src/main/assets and rebuild.",
        )
    }
}

@Composable
private fun DocumentsTab(viewModel: RagViewModel, modifier: Modifier) {
    val documents by viewModel.documents.collectAsStateWithLifecycle()
    val indexing by viewModel.indexing.collectAsStateWithLifecycle()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(viewModel::addDocument)
    }

    val documentLanguage by viewModel.documentLanguage.collectAsStateWithLifecycle()

    Column(modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Scanned pages and photos are read as:", style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(OcrLanguage.ENGLISH to "English", OcrLanguage.URDU to "اردو", OcrLanguage.MIXED to "اردو + English").forEach { (language, label) ->
                FilterChip(
                    selected = documentLanguage == language,
                    onClick = { viewModel.setDocumentLanguage(language) },
                    label = { Text(label) },
                )
            }
        }
        indexing?.let { state ->
            Card {
                Column(Modifier.padding(12.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (state.error != null) {
                        Text("Could not index ${state.name}", style = MaterialTheme.typography.titleSmall)
                        Text(state.error, color = MaterialTheme.colorScheme.error)
                        TextButton(onClick = viewModel::dismissIndexingError) { Text("Dismiss") }
                    } else {
                        Text("${state.step}: ${state.name}", style = MaterialTheme.typography.titleSmall)
                        val fraction = state.fraction
                        if (fraction == null) {
                            LinearProgressIndicator(Modifier.fillMaxWidth())
                        } else {
                            LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
                        }
                        TextButton(onClick = viewModel::cancelIndexing) { Text("Cancel") }
                    }
                }
            }
        }

        if (documents.isEmpty() && indexing == null) {
            Text("No documents yet. Add a PDF, Word file, text file or a photo of your notes to search it offline.")
        }
        LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(documents, key = { it.docId }) { document -> DocumentRow(document, onDelete = { viewModel.remove(document.docId) }) }
        }
        ExtendedFloatingActionButton(
            onClick = {
                picker.launch(
                    arrayOf(
                        "application/pdf",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "text/*",
                        "image/*",
                    ),
                )
            },
            icon = { Icon(Icons.Default.Add, contentDescription = null) },
            text = { Text("Add document") },
            modifier = Modifier.align(Alignment.End),
        )
    }
}

@Composable
private fun DocumentRow(document: DocumentInfo, onDelete: () -> Unit) {
    Card {
        Row(Modifier.padding(12.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(document.title ?: document.docId, style = MaterialTheme.typography.titleSmall)
                val status = if (document.complete) "${document.chunkCount} chunks" else "Incomplete — add it again"
                Text(status, style = MaterialTheme.typography.bodySmall)
            }
            IconButton(onClick = onDelete) { Icon(Icons.Default.Delete, contentDescription = "Remove") }
        }
    }
}

@Composable
private fun AskTab(viewModel: RagViewModel, modifier: Modifier) {
    val messages by viewModel.messages.collectAsStateWithLifecycle()
    var question by rememberSaveable { mutableStateOf("") }
    val listState = rememberLazyListState()
    val busy = messages.lastOrNull()?.streaming == true
    val voice by viewModel.voice.collectAsStateWithLifecycle()
    val voiceLanguage by viewModel.voiceLanguage.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val micPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) viewModel.startVoice() else viewModel.voicePermissionDenied()
    }

    LaunchedEffect(messages.size, messages.lastOrNull()?.answer?.length) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.lastIndex)
    }

    Column(modifier.padding(horizontal = 16.dp).imePadding()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Follow-up questions use the conversation.", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            TextButton(onClick = viewModel::newChat) { Text("New chat") }
        }
        LazyColumn(Modifier.weight(1f), state = listState, verticalArrangement = Arrangement.spacedBy(12.dp)) {
            items(messages, key = { it.id }) { MessageCard(it) }
        }
        VoicePanel(voice, voiceLanguage, viewModel)
        Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            FilledIconButton(
                onClick = {
                    when {
                        voice.listening -> viewModel.finishVoice()
                        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> viewModel.startVoice()
                        else -> micPermission.launch(Manifest.permission.RECORD_AUDIO)
                    }
                },
                enabled = !busy && !voice.transcribing,
            ) { Icon(if (voice.listening) Icons.Default.Stop else Icons.Default.Mic, contentDescription = "Ask by voice") }
            OutlinedTextField(
                value = question,
                onValueChange = { question = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask about your documents…") },
                maxLines = 4,
            )
            if (busy) {
                IconButton(onClick = viewModel::stop) { Icon(Icons.Default.Stop, contentDescription = "Stop") }
            } else {
                IconButton(
                    onClick = {
                        viewModel.ask(question)
                        question = ""
                    },
                    enabled = question.isNotBlank(),
                ) { Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Ask") }
            }
        }
    }
}

@Composable
private fun MessageCard(message: ChatMessage) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(message.question, style = MaterialTheme.typography.titleSmall)
        Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
            Column(Modifier.padding(12.dp).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                if (message.answer.isEmpty() && message.streaming) LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(message.answer)
                val notes = buildList {
                    when (message.mode) {
                        AnswerMode.EXTRACTIVE -> add("No language model on this device — showing the matching passages.")
                        AnswerMode.NO_CONTEXT -> add("Nothing relevant found.")
                        else -> Unit
                    }
                    if (message.interrupted) add("The answer was cut short.")
                    if (message.withheld > 0) add("${message.withheld} passage(s) withheld: they contained instructions aimed at the AI.")
                    if (message.invalidCitations.isNotEmpty()) {
                        add("Warning: cited source(s) ${message.invalidCitations.joinToString()} don't exist — treat that claim with care.")
                    }
                    if (message.retrievalQuery.isNotBlank() && message.retrievalQuery != message.question) {
                        add("Searched for: ${message.retrievalQuery}")
                    }
                }
                notes.forEach { Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary) }
            }
        }
        message.citations.forEach { citation ->
            Column(Modifier.padding(start = 8.dp)) {
                Text("[${citation.ordinal}] ${citation.label}", style = MaterialTheme.typography.labelMedium)
                Text(citation.snippet, style = MaterialTheme.typography.bodySmall, maxLines = 3)
                if (citation.provenance.isNotBlank()) {
                    Text(citation.provenance, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.outline)
                }
            }
        }
    }
}

@Composable
private fun ModelTab(viewModel: RagViewModel, modifier: Modifier) {
    val state by viewModel.model.collectAsStateWithLifecycle()
    Column(modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (!state.configured) {
            Text("No language model is configured.", style = MaterialTheme.typography.titleMedium)
            Text(
                "Search and extractive answers work without one. To enable generated answers, set " +
                    "demo.llmUrl, demo.llmFile, demo.llmSha256 and demo.llmSize in gradle.properties and rebuild. " +
                    "The URL must be https and the SHA-256 must match the file exactly.",
            )
            return@Column
        }
        Text(if (state.downloaded) "Model downloaded and verified" else "Model not downloaded", style = MaterialTheme.typography.titleMedium)
        state.progress?.let { progress ->
            LinearProgressIndicator(progress = { progress / 100f }, modifier = Modifier.fillMaxWidth())
            Text("$progress% — downloads only on Wi-Fi and resume if interrupted.")
            OutlinedButton(onClick = viewModel::cancelDownload) { Text("Cancel download") }
        }
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        state.deviceNote?.let { Text(it, color = MaterialTheme.colorScheme.secondary) }
        Spacer(Modifier.height(8.dp))
        if (state.downloaded) {
            OutlinedButton(onClick = viewModel::deleteModel) { Text("Delete model") }
        } else if (state.progress == null) {
            Button(onClick = viewModel::downloadModel) { Text("Download model") }
        }
    }
}

@Composable
private fun VoicePanel(voice: VoiceState, language: VoiceLanguage, viewModel: RagViewModel) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Voice:", style = MaterialTheme.typography.bodySmall)
            listOf(VoiceLanguage.URDU to "اردو", VoiceLanguage.ENGLISH to "English").forEach { (option, label) ->
                FilterChip(selected = language == option, onClick = { viewModel.setVoiceLanguage(option) }, label = { Text(label) })
            }
        }
        when {
            voice.listening -> {
                Text("Listening… stops when you pause", style = MaterialTheme.typography.bodySmall)
                LinearProgressIndicator(progress = { voice.level }, modifier = Modifier.fillMaxWidth())
                if (voice.partial.isNotBlank()) Text(voice.partial, style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = viewModel::cancelVoice) { Text("Cancel") }
            }
            voice.transcribing -> {
                Text("Understanding${voice.engine?.let { " with $it" }.orEmpty()}…", style = MaterialTheme.typography.bodySmall)
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            voice.error != null -> {
                Text(voice.error, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                if (voice.canDownloadOfflinePack) {
                    TextButton(onClick = viewModel::downloadOfflineSpeechPack) { Text("Download offline speech pack") }
                }
            }
        }
    }
}
