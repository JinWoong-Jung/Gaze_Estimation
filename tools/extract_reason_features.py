import argparse
from pathlib import Path
from typing import List, Tuple

import h5py
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
        left, sep, right = line.partition(":")
        if not sep:
            continue
        key = left.strip().lower()
        value = right.strip()
        if key == "object":
            object_text = value
        elif key in {"reason", "reasoning"}:
            reasoning_text = value

    if text_mode == "object_reasoning":
        parts = []
        if object_text:
            parts.append(object_text)
        if reasoning_text:
            parts.append(reasoning_text)
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


def build_output_key(input_root: Path, split: str, txt_path: Path) -> str:
    rel = txt_path.relative_to(input_root / split)
    return str(rel.with_suffix(""))


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
    output_root.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_root / f"{args.split}.h5"

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
    out_keys: List[str] = []

    if output_h5_path.exists() and args.overwrite:
        output_h5_path.unlink()

    with h5py.File(output_h5_path, "a") as h5f:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        if "keys" not in h5f:
            keys_ds = h5f.create_dataset("keys", shape=(0,), maxshape=(None,), dtype=str_dtype)
        else:
            keys_ds = h5f["keys"]

        if "embeddings" not in h5f:
            emb_ds = None
        else:
            emb_ds = h5f["embeddings"]

        existing_keys = set()
        if not args.overwrite and len(keys_ds) > 0:
            existing_keys = {k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in keys_ds[:]}

        def append_to_h5(batch_keys: List[str], batch_emb: torch.Tensor):
            nonlocal emb_ds, encoded
            batch_np = batch_emb.to(torch.float32).numpy()
            if emb_ds is None:
                emb_dim = int(batch_np.shape[1])
                emb_ds = h5f.create_dataset(
                    "embeddings",
                    shape=(0, emb_dim),
                    maxshape=(None, emb_dim),
                    chunks=(max(1, min(args.batch_size, 1024)), emb_dim),
                    dtype="float32",
                )
            old_n = emb_ds.shape[0]
            new_n = old_n + len(batch_keys)
            emb_ds.resize((new_n, emb_ds.shape[1]))
            keys_ds.resize((new_n,))
            emb_ds[old_n:new_n] = batch_np
            keys_ds[old_n:new_n] = batch_keys
            encoded += len(batch_keys)

        pbar = tqdm(files, desc="Extracting BGE features")
        for txt_path in pbar:
            out_key = build_output_key(input_root, args.split, txt_path)
            if (not args.overwrite) and (out_key in existing_keys):
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
            out_keys.append(out_key)

            if len(texts) >= args.batch_size:
                emb = encode_batch(model, tokenizer, texts, device, args.max_length)
                append_to_h5(out_keys, emb)
                existing_keys.update(out_keys)
                texts, out_keys = [], []

        if texts:
            emb = encode_batch(model, tokenizer, texts, device, args.max_length)
            append_to_h5(out_keys, emb)
            existing_keys.update(out_keys)

        h5f.attrs["model_name"] = args.model_name
        h5f.attrs["text_mode"] = args.text_mode
        h5f.attrs["split"] = args.split

    print("Done.")
    print(f"output_h5={output_h5_path}")
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
        default="/home/elicer/semgaze/data/gazefollow_reason/features",
        help="Output root directory for extracted .h5 embeddings (one file per split).",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument(
        "--text_mode",
        type=str,
        default="reasoning_only",
        choices=["object_reasoning", "object_only", "reasoning_only"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", help="auto, cuda, cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")
    args = parser.parse_args()
    run(args)
