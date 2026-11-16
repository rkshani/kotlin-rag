Put the embedding model here before building the demo:

  model_int8.onnx   and   vocab.txt           (English: all-MiniLM-L6-v2), or
  model_int8.onnx   and   unigram_vocab.tsv   (Urdu + English: multilingual-e5-small)

Produce them with tools/export_embedder.py. Without them the app starts and
shows these instructions instead of crashing.

OCR (optional):
  English OCR works out of the box (ML Kit is bundled).
  For Urdu, put Tesseract files in assets/tessdata/:
    tessdata/urd.traineddata   (and eng.traineddata for mixed Urdu + English pages)
  from https://github.com/tesseract-ocr/tessdata_best or tessdata_fast.

Voice (optional):
  Whisper is not shipped in the APK (it is ~250 MB). Build it with
  tools/prepare_whisper.py and push both files to the phone:
    adb push out/whisper_int8.onnx out/whisper_prompts.txt /sdcard/Android/data/io.github.rkshani.rag.demo/files/voice/
  Without it, the demo uses Android's on-device recognizer when the phone has one.
