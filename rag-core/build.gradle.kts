plugins {
    alias(libs.plugins.kotlin.jvm)
    `java-library`
}

// Target Java 17 bytecode with whatever JDK runs Gradle (AGP already requires 17+),
// same as the Android modules. A jvmToolchain(17) here would demand an installed
// JDK 17 exactly and break sync on machines that only have Android Studio's JBR 21.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
    // Turn this on in Phase 7 before publishing — it forces explicit visibility
    // and return types on every public declaration, which is what you want for a
    // library API but not while the surface is still moving.
    // explicitApi()
}

dependencies {
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform()
    testLogging { events("passed", "skipped", "failed") }
}
