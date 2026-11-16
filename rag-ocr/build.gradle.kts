plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "io.github.rkshani.rag.ocr"
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
    api(project(":rag-loader-pdf"))
    // Latin OCR, model bundled in the APK (~4 MB per ABI): works offline from first launch.
    implementation(libs.mlkit.text.recognition)
    // Urdu (and any other Tesseract language). Ship the .traineddata files yourself.
    implementation(libs.tesseract4android)
    implementation(libs.androidx.exifinterface)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.junit4)
}
