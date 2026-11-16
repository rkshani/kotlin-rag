pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Tesseract4Android is published on JitPack only.
        maven("https://jitpack.io") {
            content { includeGroup("cz.adaptech.tesseract4android") }
        }
    }
}

rootProject.name = "kotlin-rag"

include(":rag-core")
include(":rag-embed-onnx")
include(":rag-store-room")
include(":rag-loader-pdf")
include(":rag-ocr")
include(":rag-voice")
include(":rag-llm-litertlm")
include(":demo")

