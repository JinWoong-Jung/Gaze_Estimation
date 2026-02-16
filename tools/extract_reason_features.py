import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def import_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer

        return AutoModel, AutoTokenizer
    except Exception as exc:
        # Some `transformers` versions import `sklearn` (and thus `scipy`) at import-time for generation code.
        # For embedding-only usage, we can safely stub `sklearn.metrics.roc_curve` to avoid hard failures when
        # sklearn/scipy wheels are ABI-incompatible with the local NumPy.
        msg = str(exc)
        needs_sklearn_stub = any(
            needle in msg
            for needle in (
                "sklearn",
                "scipy",
                "numpy.core.multiarray failed to import",
                "compiled using NumPy 1.x",
            )
        )
        if not needs_sklearn_stub:
            raise

        import sys
        import types

        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")

        def roc_curve(*_args, **_kwargs):
            raise RuntimeError("roc_curve is unavailable (sklearn stubbed for embedding-only use).")

        metrics_stub.roc_curve = roc_curve
        sklearn_stub.metrics = metrics_stub

        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub

        from transformers import AutoModel, AutoTokenizer

        return AutoModel, AutoTokenizer


def parse_reason_text(raw_text: str, text_mode: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    object_text = ""
    reasoning_text = ""
    for line in lines:
        lower = line.lower()
        if lower.startswith("object:"):
            object_text = line.split(":", 1)[1].strip()
        elif lower.startswith("reasoning:"):
            reasoning_text = line.split(":", 1)[1].strip()

    if text_mode == "object_reasoning":
        parts = []
        if object_text:
            parts.append(f"Object: {object_text}")
        if reasoning_text:
            parts.append(f"Reasoning: {reasoning_text}")
        if parts:
            return "\n".join(parts)
    elif text_mode == "object_only":
        if object_text:
            return object_text
    elif text_mode == "reasoning_only":
        if reasoning_text:
            return reasoning_text
    else:
        raise ValueError(f"Unsupported text_mode: {text_mode}")

    # Fallback if expected tagged fields are missing.
    return "\n".join(lines)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled


@torch.inference_mode()
def encode_batch(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu()


def gather_files(input_root: Path, split: str) -> List[Path]:
    split_dir = input_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return sorted(split_dir.rglob("*.txt"))


def build_output_path(output_root: Path, input_root: Path, split: str, txt_path: Path) -> Path:
    rel = txt_path.relative_to(input_root / split)
    return output_root / split / rel.with_suffix(".pt")


def run(args):
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    # Convenience: allow passing the dataset root (containing `output/`) instead of the `output/` directory.
    if (input_root / "output").exists() and not (input_root / args.split).exists():
        input_root = input_root / "output"

    # Convenience: allow passing a parent root instead of the `features/` directory.
    # - `.../gazefollow_reason` -> `.../gazefollow_reason/features`
    # - `.../gazefollow_reason_bge` -> `.../gazefollow_reason_bge/features` (legacy)
    if output_root.name in {"gazefollow_reason", "gazefollow_reason_bge"}:
        output_root = output_root / "features"

    files = gather_files(input_root, args.split)
    print(f"Found {len(files)} txt files for split={args.split}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    AutoModel, AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()
    print(f"Using model={args.model_name} on device={device} text_mode={args.text_mode}")

    skipped_existing = 0
    failed = 0
    encoded = 0
    texts: List[str] = []
    out_paths: List[Path] = []

    pbar = tqdm(files, desc="Extracting BGE features")
    for txt_path in pbar:
        out_path = build_output_path(output_root, input_root, args.split, txt_path)
        if out_path.exists() and (not args.overwrite):
            skipped_existing += 1
            continue

        try:
            raw_text = txt_path.read_text(encoding="utf-8")
            text = parse_reason_text(raw_text, args.text_mode)
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed reading/parsing {txt_path}: {exc}")
            continue

        texts.append(text)
        out_paths.append(out_path)

        if len(texts) >= args.batch_size:
            emb = encode_batch(model, tokenizer, texts, device, args.max_length)
            for feature, save_path in zip(emb, out_paths):
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feature.to(torch.float32), save_path)
                encoded += 1
            texts, out_paths = [], []

    if texts:
        emb = encode_batch(model, tokenizer, texts, device, args.max_length)
        for feature, save_path in zip(emb, out_paths):
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feature.to(torch.float32), save_path)
            encoded += 1

    print("Done.")
    print(f"encoded={encoded}")
    print(f"skipped_existing={skipped_existing}")
    print(f"failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BGE features from gaze reasoning txt files.")
    parser.add_argument(
        "--input_root",
        type=str,
        default="/home/elicer/semgaze/data/bucket_data/data/gazefollow_reason/output",
        help="Root directory containing reason txt files by split, e.g. <input_root>/train/**/*.txt",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/elicer/semgaze/data/bucket_data/data/gazefollow_reason/features",
        help="Output root directory for extracted .pt embeddings.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument(
        "--text_mode",
        type=str,
        default="object_reasoning",
        choices=["object_reasoning", "object_only", "reasoning_only"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    args = parser.parse_args()
    run(args)
