# ONNX Runtime binds its Java classes from native code (JNI). R8 cannot see
# those references, so without these rules a minified release build fails at
# session creation while debug builds work.
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * { native <methods>; }
