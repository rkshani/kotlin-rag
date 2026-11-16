package io.github.rkshani.rag.ocr

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.googlecode.tesseract.android.TessBaseAPI
import java.io.File

/** Text from one image. Blocking — run it on a background thread. */
interface OcrEngine : AutoCloseable {
    fun supports(language: OcrLanguage): Boolean

    fun recognize(bitmap: Bitmap, language: OcrLanguage): String
}

/**
 * Google ML Kit text recognition, Latin script, model bundled in the APK.
 * Fast and accurate for English; it has no Arabic-script model, so Urdu goes to
 * Tesseract instead.
 */
class MlKitOcr : OcrEngine {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    override fun supports(language: OcrLanguage): Boolean = language == OcrLanguage.ENGLISH

    override fun recognize(bitmap: Bitmap, language: OcrLanguage): String {
        require(supports(language)) { "ML Kit cannot read $language" }
        // Tasks.await blocks; the loader already runs off the main thread.
        return Tasks.await(recognizer.process(InputImage.fromBitmap(bitmap, 0))).text
    }

    override fun close() = recognizer.close()
}

/**
 * Tesseract 5 through Tesseract4Android. Handles Urdu (`urd`), and mixed
 * Urdu + English pages (`urd+eng`).
 *
 * Quality note: the stock `urd` model reads Naskh-style print well. Nastaliq —
 * most Urdu books and newspapers — comes out much worse; that needs a model
 * trained on Nastaliq, which this class accepts like any other traineddata.
 *
 * One TessBaseAPI per language code, created on first use. Tesseract is not
 * thread-safe, so recognition is serialised.
 *
 * @param dataDir the directory that CONTAINS `tessdata/` (Tesseract's convention).
 *   [TessData.install] prepares it from assets.
 */
class TesseractOcr(
    private val dataDir: File,
    private val pageSegMode: Int = TessBaseAPI.PageSegMode.PSM_AUTO,
) : OcrEngine {

    private val engines = HashMap<String, TessBaseAPI>()

    override fun supports(language: OcrLanguage): Boolean =
        TessDataNames.required(language.tesseractCode).all { File(File(dataDir, "tessdata"), it).isFile }

    @Synchronized
    override fun recognize(bitmap: Bitmap, language: OcrLanguage): String {
        if (!supports(language)) {
            throw OcrUnavailableException("missing ${TessDataNames.required(language.tesseractCode)} in ${File(dataDir, "tessdata")}")
        }
        val api = engines.getOrPut(language.tesseractCode) {
            TessBaseAPI().also { api ->
                check(api.init(dataDir.absolutePath, language.tesseractCode)) {
                    "Tesseract could not load '${language.tesseractCode}'"
                }
                api.setPageSegMode(pageSegMode)
            }
        }
        return try {
            api.setImage(bitmap)
            api.getUTF8Text().orEmpty()
        } finally {
            api.clear()
        }
    }

    @Synchronized
    override fun close() {
        engines.values.forEach { it.recycle() }
        engines.clear()
    }
}

/**
 * Sends each language to the engine that reads it best: English to ML Kit,
 * Urdu and mixed pages to Tesseract. Either engine may be absent.
 */
class RoutingOcr(
    private val latin: OcrEngine? = MlKitOcr(),
    private val tesseract: OcrEngine? = null,
) : OcrEngine {

    override fun supports(language: OcrLanguage): Boolean = pick(language) != null

    override fun recognize(bitmap: Bitmap, language: OcrLanguage): String {
        val engine = pick(language)
            ?: throw OcrUnavailableException("no OCR engine for $language — add the Tesseract traineddata for it")
        return engine.recognize(bitmap, language)
    }

    private fun pick(language: OcrLanguage): OcrEngine? = when (language) {
        OcrLanguage.ENGLISH -> latin?.takeIf { it.supports(language) } ?: tesseract?.takeIf { it.supports(language) }
        OcrLanguage.URDU, OcrLanguage.MIXED -> tesseract?.takeIf { it.supports(language) }
    }

    override fun close() {
        runCatching { latin?.close() }
        runCatching { tesseract?.close() }
    }
}

/**
 * Copies `.traineddata` files shipped in the APK's assets into app storage,
 * where Tesseract can open them (it needs real files, not asset streams).
 */
object TessData {

    /**
     * @param assetDir assets folder holding e.g. `urd.traineddata`, `eng.traineddata`.
     * @return the directory to pass to [TesseractOcr] (it contains `tessdata/`).
     */
    fun install(context: Context, assetDir: String = "tessdata"): File {
        val root = File(context.filesDir, "ocr")
        val target = File(root, "tessdata").apply { mkdirs() }
        context.assets.list(assetDir).orEmpty()
            .filter { TessDataNames.isValid(it) }
            .forEach { name ->
                val out = File(target, name)
                // Size check catches a half-copied file from an earlier crash.
                val expected = runCatching { context.assets.openFd("$assetDir/$name").use { it.length } }.getOrNull()
                if (out.isFile && (expected == null || out.length() == expected)) return@forEach
                val partial = File(target, "$name.part")
                context.assets.open("$assetDir/$name").use { input -> partial.outputStream().use { input.copyTo(it) } }
                if (!partial.renameTo(out)) error("could not install $name")
            }
        return root
    }
}
