plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "io.github.rkshani.rag.voice"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
    }

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
    api(project(":rag-core"))
    implementation(libs.onnxruntime.android)
    // Audio decoding, log-mel features, beam search and BPE decoding inside the
    // all-in-one Whisper model are onnxruntime-extensions custom ops.
    implementation(libs.onnxruntime.extensions.android)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.junit4)
}
