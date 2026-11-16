#!/usr/bin/env python3
"""
Prepares an offline Whisper model for rag-voice (WhisperOnnx / WhisperVoiceInput).

rag-voice runs the "all-in-one" Whisper ONNX model built by Microsoft Olive's
whisper example: audio pre-processing, encoder, decoder with beam search and
text decoding in one graph. Building it is Olive's job; this script drives it,
checks the result, and writes the language prompt file the app needs.

1. Build (needs a machine with internet; takes a while):

    git clone https://github.com/microsoft/Olive
    pip install -r requirements.txt   # this folder's, plus Olive's examples/whisper/requirements.txt
    python prepare_whisper.py --olive-example-dir Olive/examples/whisper --model openai/whisper-small

   This runs Olive with --multilingual (required for Urdu: it adds the
   decoder_input_ids input that forces the language) and --no_audio_decoder
   (the app sends raw 16 kHz samples, so no mp3 decoder is needed).

2. Or check a model you already built:

    python prepare_whisper.py --onnx path/to/whisper_cpu_int8_model.onnx --model openai/whisper-small

Either way it writes into ./out:
    whisper_int8.onnx      the model to copy to the phone
    whisper_prompts.txt    forced prompt ids per language (en, ur)

Optionally transcribe a test recording with --test-wav question.wav (16 kHz mono).

Model size for Urdu: tiny and base are poor; small (~250 MB int8) is the
practical minimum on a phone.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import wave

import numpy as np

LANGUAGES = {"en": "english", "ur": "urdu"}


def run_olive(example_dir: str, model: str) -> str:
    example_dir = os.path.abspath(example_dir)
    if not os.path.isfile(os.path.join(example_dir, "prepare_whisper_configs.py")):
        raise SystemExit(f"{example_dir} is not Olive's examples/whisper folder")
    config = "whisper_cpu_int8.json"
    steps = [
        [sys.executable, "prepare_whisper_configs.py", "--model_name", model, "--multilingual", "--no_audio_decoder"],
        ["olive", "run", "--config", config, "--setup"],
        ["olive", "run", "--config", config],
    ]
    for step in steps:
        print("$", " ".join(step))
        subprocess.run(step, cwd=example_dir, check=True)
    candidates = sorted(
        glob.glob(os.path.join(example_dir, "**", "*.onnx"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    candidates = [c for c in candidates if "int8" in os.path.basename(c) or "model" in os.path.basename(c)]
    if not candidates:
        raise SystemExit("Olive finished but no .onnx model was found — check its output folder")
    print("built:", candidates[0])
    return candidates[0]


def inspect(model_path: str):
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    options = ort.SessionOptions()
    options.register_custom_ops_library(get_library_path())
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    inputs = {i.name: (i.type, i.shape) for i in session.get_inputs()}
    print("model inputs:")
    for name, (kind, shape) in inputs.items():
        print(f"  {name:22s} {kind:16s} {shape}")
    if "audio_pcm" not in inputs and "audio_stream" not in inputs:
        raise SystemExit("not an all-in-one Whisper model (no audio_pcm / audio_stream input)")
    if "decoder_input_ids" not in inputs:
        print("WARNING: no decoder_input_ids input — this export is English-only and cannot be forced to Urdu.")
        print("         Rebuild with --multilingual.")
    return session, inputs


def write_prompts(model: str, out_dir: str) -> dict:
    from transformers import WhisperConfig, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model)
    config = WhisperConfig.from_pretrained(model)
    prompts = {}
    for code, name in LANGUAGES.items():
        forced = processor.get_decoder_prompt_ids(language=name, task="transcribe", no_timestamps=True)
        ids = [config.decoder_start_token_id] + [token for _, token in forced]
        prompts[code] = ids
    path = os.path.join(out_dir, "whisper_prompts.txt")
    with open(path, "w") as f:
        f.write(f"# forced decoder prompts for {model}: <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>\n")
        for code, ids in prompts.items():
            f.write(code + " " + " ".join(str(i) for i in ids) + "\n")
    print(f"wrote {path}: {prompts}")
    return prompts


def transcribe(session, inputs: dict, wav_path: str, prompt) -> str:
    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit("test wav must be 16 kHz, mono, 16-bit")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    feed = {
        "max_length": np.array([200], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "num_beams": np.array([1], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
    }
    if "audio_pcm" in inputs:
        feed["audio_pcm"] = audio[np.newaxis, :]
    else:
        with open(wav_path, "rb") as f:
            feed["audio_stream"] = np.frombuffer(f.read(), dtype=np.uint8)[np.newaxis, :]
    if "attention_mask" in inputs:
        feed["attention_mask"] = np.zeros((1, 80, 3000), dtype=np.int32)
    if "decoder_input_ids" in inputs:
        feed["decoder_input_ids"] = np.array([prompt], dtype=np.int32)
    feed = {k: v for k, v in feed.items() if k in inputs}
    return session.run(None, feed)[0][0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-small", help="Hugging Face model id (for the prompt ids)")
    parser.add_argument("--olive-example-dir", help="path to Olive/examples/whisper — builds the model")
    parser.add_argument("--onnx", help="an already built all-in-one model")
    parser.add_argument("--out", default="out")
    parser.add_argument("--test-wav", help="16 kHz mono 16-bit wav to transcribe as a smoke test")
    parser.add_argument("--test-language", default="ur", choices=sorted(LANGUAGES))
    args = parser.parse_args()

    if not args.onnx and not args.olive_example_dir:
        parser.error("give --olive-example-dir to build, or --onnx to check an existing model")

    model_path = args.onnx or run_olive(args.olive_example_dir, args.model)
    os.makedirs(args.out, exist_ok=True)
    session, inputs = inspect(model_path)
    prompts = write_prompts(args.model, args.out)

    target = os.path.join(args.out, "whisper_int8.onnx")
    if os.path.abspath(model_path) != os.path.abspath(target):
        shutil.copyfile(model_path, target)
    print(f"copied model to {target} ({os.path.getsize(target) / 1e6:.0f} MB)")

    if args.test_wav:
        text = transcribe(session, inputs, args.test_wav, prompts[args.test_language])
        print(f"transcript ({args.test_language}): {text}")

    print("\nnext: push whisper_int8.onnx and whisper_prompts.txt to the phone, e.g.")
    print("  adb push out/whisper_int8.onnx out/whisper_prompts.txt /sdcard/Android/data/io.github.rkshani.rag.demo/files/voice/")


if __name__ == "__main__":
    main()
