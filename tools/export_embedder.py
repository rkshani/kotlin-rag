#!/usr/bin/env python3
"""
Exports a sentence-transformers model to quantized ONNX and generates the
parity fixtures the Android tests check against.

Run in Colab or any machine with internet (the phone never does this):

    pip install -r requirements.txt
    python export_embedder.py --model sentence-transformers/all-MiniLM-L6-v2

Outputs into ./out:
    model.onnx           fp32 export
    model_int8.onnx      dynamic int8 — this is what ships
    vocab.txt            WordPiece vocab            (BERT-style models), or
    unigram_vocab.tsv    SentencePiece Unigram vocab (XLM-R models: multilingual-e5, mMiniLM)
    parity.json          reference ids + embeddings for the Android test

Copy model_int8.onnx and the vocab file into the app, and parity.json into
rag-embed-onnx/src/androidTest/assets/.

Urdu needs a multilingual model:

    python export_embedder.py --model intfloat/multilingual-e5-small

A cross-encoder for OnnxCrossEncoderReranker:

    python export_embedder.py --cross-encoder --model cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import argparse
import json
import os
import shutil

import numpy as np

# Kept deliberately diverse: accents, NBSP, zero-width, CJK, Urdu, punctuation,
# a >100 char word, digits. These are the inputs that break tokenizer ports.
PARITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is unaffable!",
    "CAFÉ naïve résumé",
    "don't  stop believing",
    "hello,world (2024) 50% off",
    "北京 hello 한글 mixed script",
    "یہ اردو کا جملہ ہے۔",
    "Hello\u00a0World\u200b!",
    "a" * 150,
    "Retrieval augmented generation runs entirely on device.",
    "Machine learning models can be quantized to int8 for mobile inference.",
    "short",
    "شمسی توانائی سے بجلی بنتی ہے اور بیٹری رات کے لیے محفوظ کرتی ہے۔",
    "کِتَاب پر زیر زبر — diacritics",
]


def write_tokenizer_files(out_dir: str) -> str:
    """Writes the vocab file the Kotlin side reads. Returns the TokenizerFormat name."""
    tok_json = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(os.path.join(out_dir, "vocab.txt")):
        return "WORDPIECE_VOCAB"
    if not os.path.exists(tok_json):
        raise SystemExit("no vocab.txt and no tokenizer.json — unsupported tokenizer")
    with open(tok_json, encoding="utf-8") as f:
        spec = json.load(f)
    model = spec["model"]
    if model.get("type") != "Unigram":
        raise SystemExit(f"tokenizer type {model.get('type')} is not supported (WordPiece or Unigram only)")
    specials = {t["id"]: t["content"] for t in spec.get("added_tokens", []) if t.get("special")}
    unk_id = model.get("unk_id")
    path = os.path.join(out_dir, "unigram_vocab.tsv")
    with open(path, "w", encoding="utf-8") as f:
        for i, (piece, score) in enumerate(model["vocab"]):
            if i == unk_id:
                kind = "UNKNOWN"
            elif i in specials:
                kind = "CONTROL"
            else:
                kind = "NORMAL"
            text = piece.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{text}\t{score}\t{kind}\n")
    print(f"wrote {path} ({len(model['vocab'])} pieces)")
    return "UNIGRAM_TSV"


def export(model_id: str, out_dir: str, seq_len: int) -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)

    print(f"exporting {model_id} -> onnx")
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)

    fp32 = os.path.join(out_dir, "model.onnx")
    int8 = os.path.join(out_dir, "model_int8.onnx")

    print("quantizing to int8 (dynamic)")
    # Per-channel weights keep more accuracy on attention projections at no
    # meaningful size cost. QUInt8 is the safe activation type on ARM.
    quantize_dynamic(
        model_input=fp32,
        model_output=int8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"MatMulConstBOnly": True},
    )

    for path in (fp32, int8):
        print(f"  {os.path.basename(path)}: {os.path.getsize(path) / 1e6:.1f} MB")

    tokenizer_format = write_tokenizer_files(out_dir)
    build_parity(model_id, out_dir, int8, seq_len, tokenizer_format)


def build_parity(model_id: str, out_dir: str, int8_path: str, seq_len: int, tokenizer_format: str) -> None:
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    reference = SentenceTransformer(model_id)

    print("computing reference embeddings (fp32, sentence-transformers)")
    st_vectors = reference.encode(PARITY_TEXTS, normalize_embeddings=True)

    print("computing int8 ONNX embeddings")
    session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}

    entries = []
    max_drift = 0.0
    for text, st_vec in zip(PARITY_TEXTS, st_vectors):
        encoded = tokenizer(
            text, truncation=True, max_length=seq_len, return_tensors="np"
        )
        feed = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])

        hidden = session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        vec = pooled[0] / np.linalg.norm(pooled[0])

        drift = float(np.dot(vec, st_vec))
        max_drift = max(max_drift, 1.0 - drift)

        entries.append(
            {
                "text": text,
                "tokens": tokenizer.tokenize(text)[: seq_len - 2],
                "ids": encoded["input_ids"][0].tolist(),
                "embedding": [round(float(x), 6) for x in vec],
            }
        )

    print(f"int8 vs fp32 worst-case cosine drift: {max_drift:.5f}")
    if max_drift > 0.02:
        print("  WARNING: quantization drift is high. Try per_channel=True or fp16 instead.")

    payload = {
        "model": model_id,
        "dimensions": len(entries[0]["embedding"]),
        "max_sequence_tokens": seq_len,
        "pooling": "mean",
        "tokenizer": tokenizer_format,
        "lowercase": bool(getattr(tokenizer, "do_lower_case", False)),
        "entries": entries,
    }
    parity_path = os.path.join(out_dir, "parity.json")
    with open(parity_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"wrote {parity_path} ({os.path.getsize(parity_path) / 1e3:.0f} KB)")


def export_cross_encoder(model_id: str, out_dir: str, seq_len: int) -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer
    import onnxruntime as ort

    os.makedirs(out_dir, exist_ok=True)
    print(f"exporting cross-encoder {model_id} -> onnx")
    ORTModelForSequenceClassification.from_pretrained(model_id, export=True).save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)
    fp32 = os.path.join(out_dir, "model.onnx")
    int8 = os.path.join(out_dir, "model_int8.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True)
    write_tokenizer_files(out_dir)

    # Sanity check: the relevant passage must outscore the irrelevant one.
    session = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}
    enc = tokenizer(["how do batteries store solar energy"] * 2,
                    ["Batteries store extra solar energy for use at night.", "Biryani is steamed in a sealed pot."],
                    truncation=True, max_length=seq_len, padding=True, return_tensors="np")
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in names}
    logits = session.run(None, feed)[0]
    print("relevant vs irrelevant logits:", logits.ravel().tolist())
    if logits.ravel()[0] <= logits.ravel()[-1]:
        print("  WARNING: the relevant passage did not score higher — check the model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", default="out")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--cross-encoder", action="store_true", help="export a reranker instead of an embedder")
    args = parser.parse_args()

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)

    if args.cross_encoder:
        export_cross_encoder(args.model, args.out, args.seq_len)
        return
    export(args.model, args.out, args.seq_len)
    print("\nnext: copy model_int8.onnx + vocab.txt to the app, parity.json to androidTest/assets/")


if __name__ == "__main__":
    main()
