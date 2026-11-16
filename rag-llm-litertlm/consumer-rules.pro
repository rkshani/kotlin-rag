# LiteRT-LM calls back into these classes from native code (JNI). R8 cannot see
# those references, so without these rules a minified release build crashes on
# the first generation while debug builds work.
-keep class com.google.ai.edge.litertlm.** { *; }
-keepclasseswithmembernames class * { native <methods>; }

# WorkManager instantiates the download worker reflectively by class name.
-keep class io.github.rkshani.rag.llm.ModelDownloadWorker { <init>(...); }
