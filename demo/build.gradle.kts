plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

// Optional on-device LLM for generated answers. Without these the demo still
// works and answers extractively. Put them in ~/.gradle/gradle.properties or
// pass -P on the command line; never commit a model URL with a token in it.
fun prop(name: String) = (findProperty(name) as String?).orEmpty()

android {
    namespace = "io.github.rkshani.rag.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.rkshani.rag.demo"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a") }

        buildConfigField("String", "LLM_URL", "\"${prop("demo.llmUrl")}\"")
        buildConfigField("String", "LLM_FILE", "\"${prop("demo.llmFile")}\"")
        buildConfigField("String", "LLM_SHA256", "\"${prop("demo.llmSha256")}\"")
        buildConfigField("long", "LLM_SIZE", "${prop("demo.llmSize").ifBlank { "0" }}L")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    // Models ship in assets; compressing them only slows startup, and
    // traineddata must be uncompressed for the installer's size check.
    androidResources { noCompress += listOf("onnx", "tsv", "traineddata") }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(project(":rag-core"))
    implementation(project(":rag-embed-onnx"))
    implementation(project(":rag-store-room"))
    implementation(project(":rag-loader-pdf"))
    implementation(project(":rag-ocr"))
    implementation(project(":rag-voice"))
    implementation(project(":rag-llm-litertlm"))

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
}
