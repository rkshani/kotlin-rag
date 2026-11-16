# Tesseract4Android's native code calls back into TessBaseAPI (progress, fields).
-keep class com.googlecode.tesseract.android.** { *; }
-keepclasseswithmembernames class * { native <methods>; }
