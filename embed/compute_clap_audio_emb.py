#!/usr/bin/env python3
"""
Compute & save CLAP audio embeddings (batched, resumable, memory-friendly).

Resolution rule (explicit & deterministic):
  1) Try metadata PATH joined with audio_root: audio_root / PATH
  2) Try audio_root / Path(PATH).name                 (files might be stored flat)
  3) Try audio_root / first2(PATH) / <numeric>.mp3    (numeric from TRACK_ID, keeps leading zeros)
  4) Try audio_root / first2(PATH) / <int(numeric)>.mp3 (trim leading zeros)

Embeddings:
  - Uses clap.get_audio_embedding_from_filelist(x=list_of_paths, use_tensor=False)
  - Creates/updates a numpy.memmap clap_audio_embs.npy and a boolean checkpoint clap_audio_processed.npy

Outputs (in out-dir):
  - clap_audio_embs.npy        (memmap shape N x D)
  - clap_audio_processed.npy   (boolean processed flags)
  - clap_audio_missing.json    (rows where no candidate existed)
  - clap_audio_failed_loads.json
  - clap_audio_failed_embeddings.json
  - resolved_paths.csv         (index, TRACK_ID, resolved_path, matched_candidate_index)
"""
import argparse
import json
import time
import traceback
from pathlib import Path
import re
import csv

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import gc

# optional imports
try:
    from laion_clap import CLAP_Module
except Exception:
    CLAP_Module = None

try:
    import librosa
except Exception:
    librosa = None


# helpers
def ensure_libs():
    if CLAP_Module is None:
        raise RuntimeError("laion_clap not available. Install laion_clap in your environment.")
    if librosa is None:
        raise RuntimeError("librosa not available. Install librosa and soundfile: pip install librosa soundfile")


def numeric_from_track(track_id: str) -> str:
    s = str(track_id)
    if "_" in s:
        candidate = s.split("_")[-1]
        if candidate.isdigit():
            return candidate
    m = re.findall(r"(\d+)", s)
    return m[-1] if m else s


def build_candidate_list(audio_root: Path, path_field: str, numeric: str, try_trim: bool = True):
    """
    Return ordered candidate paths (strings).
    Index order corresponds to preference.
    """
    candidates = []
    # normalize incoming PATH: allow both slashes/backslashes
    p = Path(path_field)
    # candidate 0: audio_root / PATH (preserves subfolders)
    candidates.append(audio_root / p)
    # candidate 1: audio_root / filename only
    candidates.append(audio_root / p.name)
    # candidate 2: audio_root / first2(path_field) / numeric (preserve zeros)
    folder = (path_field or "")[:2]
    if folder:
        candidates.append(audio_root / folder / f"{numeric}.mp3")
    else:
        candidates.append(audio_root / f"{numeric}.mp3")
    # candidate 3: trimmed numeric
    if try_trim and numeric.isdigit():
        trimmed = str(int(numeric))
        if folder:
            candidates.append(audio_root / folder / f"{trimmed}.mp3")
        else:
            candidates.append(audio_root / f"{trimmed}.mp3")
    # ensure string paths and uniqueness (preserve order)
    out = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def main(metadata_csv: str,
         audio_root: str,
         out_dir: str,
         batch_size: int = 32,
         use_gpu: bool = True,
         clap_ckpt: str = None,
         clap_amodel: str = "HTSAT-base",
         sr: int = 48000,
         try_trim_numeric: bool = True):
    ensure_libs()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root = Path(audio_root)

    meta = pd.read_csv(metadata_csv, dtype=str).fillna("")
    if "TRACK_ID" not in meta.columns or "PATH" not in meta.columns:
        raise RuntimeError("metadata CSV must contain TRACK_ID and PATH columns")

    n = len(meta)
    print(f"[INFO] Loaded metadata rows: {n}")

    # Build candidate lists and try to resolve to an existing file (using ordered candidates)
    candidate_lists = []
    resolved = ["" for _ in range(n)]
    matched_candidate_idx = [-1 for _ in range(n)]
    missing_rows = []

    for idx, row in meta.iterrows():
        numeric = numeric_from_track(row["TRACK_ID"])
        path_field = row["PATH"]
        cands = build_candidate_list(audio_root, path_field, numeric, try_trim=try_trim_numeric)
        candidate_lists.append(cands)
        found = None
        found_idx = -1
        for i_c, cand in enumerate(cands):
            if Path(cand).is_file():
                found = str(Path(cand).resolve())
                found_idx = i_c
                break
        if found:
            resolved[idx] = found
            matched_candidate_idx[idx] = found_idx
        else:
            missing_rows.append({"index": int(idx), "TRACK_ID": row["TRACK_ID"], "path_guesses": cands})

    print(f"[INFO] Resolved files: {sum(1 for p in resolved if p)}; missing guesses: {len(missing_rows)}")
    # save missing guesses
    with open(out_dir / "clap_audio_missing.json", "w", encoding="utf-8") as f:
        json.dump(missing_rows, f, indent=2, ensure_ascii=False)

    # Save resolved mapping for auditing
    resolved_csv = out_dir / "resolved_paths.csv"
    with open(resolved_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "TRACK_ID", "resolved_path", "matched_candidate_index"])
        for i, track in enumerate(meta["TRACK_ID"]):
            writer.writerow([i, track, resolved[i], matched_candidate_idx[i]])

    # Initialize CLAP module
    device = torch.device("cuda:0") if (torch.cuda.is_available() and use_gpu) else torch.device("cpu")
    print(f"[INFO] Initializing CLAP model on {device} with amodel={clap_amodel} ...")
    clap = CLAP_Module(enable_fusion=False, device=device, amodel=clap_amodel)
    if clap_ckpt:
        print("[INFO] Loading CLAP checkpoint:", clap_ckpt)
        clap.load_ckpt(ckpt=clap_ckpt)

    # Ensure method exists
    if not hasattr(clap, "get_audio_embedding_from_filelist"):
        # if not present, try get_audio_embedding_from_data later - but prefer filelist for speed
        raise RuntimeError("CLAP_Module missing get_audio_embedding_from_filelist. Inspect dir(clap).")

    # determine embedding dim by calling on one sample file (the first resolved)
    sample_idx = None
    for i, rp in enumerate(resolved):
        if rp:
            sample_idx = i
            break
    if sample_idx is None:
        raise RuntimeError("No audio files found with the current path resolution. Check clap_audio_missing.json")

    test_path = resolved[sample_idx]
    print("[INFO] Testing CLAP embedding call on sample:", test_path)
    try:
        emb_sample = clap.get_audio_embedding_from_filelist(x=[test_path], use_tensor=False)
    except Exception as e:
        raise RuntimeError(f"CLAP get_audio_embedding_from_filelist call failed on sample: {e}")
    emb_sample = np.asarray(emb_sample)
    if emb_sample.ndim == 2:
        emb_dim = emb_sample.shape[1]
    elif emb_sample.ndim == 1:
        emb_dim = emb_sample.shape[0]
    else:
        raise RuntimeError(f"Unexpected embedding shape: {emb_sample.shape}")
    print(f"[INFO] CLAP audio embedding dimension = {emb_dim}")

    # Prepare memmap and processed checkpoint
    embs_path = out_dir / "clap_audio_embs.npy"
    proc_path = out_dir / "clap_audio_processed.npy"
    failed_loads_path = out_dir / "clap_audio_failed_loads.json"
    failed_emb_path = out_dir / "clap_audio_failed_embeddings.json"

    if embs_path.exists() and proc_path.exists():
        print("[INFO] Found existing memmap + checkpoint -> resuming.")
        embs = np.memmap(str(embs_path), dtype=np.float32, mode="r+", shape=(n, emb_dim))
        processed = np.load(proc_path)
        if processed.shape[0] != n:
            raise RuntimeError("Existing checkpoint length mismatch.")
    else:
        print("[INFO] Creating new memmap and checkpoint.")
        embs = np.memmap(str(embs_path), dtype=np.float32, mode="w+", shape=(n, emb_dim))
        embs[:] = 0.0
        processed = np.zeros(n, dtype=bool)
        np.save(proc_path, processed)

    failed_loads = []
    failed_embeddings = []

    # Build list of to-process (index,resolved_path) where resolved_path exists and not yet processed
    to_process = [(i, resolved[i]) for i in range(n) if (resolved[i] and (not processed[i]))]

    print(f"[INFO] To process (unprocessed resolved files): {len(to_process)}")

    # optional test load function (to detect broken files before calling CLAP)
    def test_load(path):
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)

    pbar = tqdm(range(0, len(to_process), batch_size), desc="audio_batches")
    for start in pbar:
        batch = to_process[start:start+batch_size]
        batch_idxs = [x[0] for x in batch]
        batch_paths = [x[1] for x in batch]

        # check readability with librosa (we call CLAP by filelist, but it's useful to detect broken files beforehand)
        good_paths = []
        good_idxs = []
        for bi, bp in zip(batch_idxs, batch_paths):
            try:
                _ = test_load(bp)
                good_paths.append(bp)
                good_idxs.append(bi)
            except Exception as e:
                failed_loads.append({"index": int(bi), "TRACK_ID": meta.loc[bi, "TRACK_ID"], "path": bp, "error": str(e),
                                     "traceback": traceback.format_exc()})
                print(f"[WARN] librosa failed to read {bp}: {e}")

        if len(good_paths) == 0:
            continue

        # call CLAP on filelist
        try:
            emb_batch = clap.get_audio_embedding_from_filelist(x=good_paths, use_tensor=False)
            emb_batch = np.asarray(emb_batch).astype(np.float32)
        except Exception as e:
            # fallback to per-file (isolate failures)
            print(f"[WARN] Batch CLAP call failed: {e}. Trying per-file.")
            for i_local, idx_global in enumerate(good_idxs):
                path_single = good_paths[i_local]
                try:
                    emb_single = clap.get_audio_embedding_from_filelist(x=[path_single], use_tensor=False)
                    emb_single = np.asarray(emb_single).astype(np.float32)
                    if emb_single.ndim == 1:
                        emb_single = emb_single.reshape(1, -1)
                    embs[idx_global, :] = l2_normalize_np(emb_single)[0]
                    processed[idx_global] = True
                except Exception as e_single:
                    failed_embeddings.append({"index": int(idx_global), "TRACK_ID": meta.loc[idx_global, "TRACK_ID"],
                                               "path": path_single, "error": str(e_single),
                                               "traceback": traceback.format_exc()})
            embs.flush()
            np.save(proc_path, processed)
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
            continue

        # normalize + write
        if emb_batch.ndim == 1 and len(good_idxs) == 1:
            emb_batch = emb_batch.reshape(1, -1)
        if emb_batch.ndim == 3:
            emb_batch = emb_batch.reshape(emb_batch.shape[0], -1)
        if emb_batch.shape[0] != len(good_idxs):
            # safety check
            raise RuntimeError(f"Embedding count mismatch: {emb_batch.shape[0]} vs {len(good_idxs)}")

        emb_batch = l2_normalize_np(emb_batch)
        for i_local, idx_global in enumerate(good_idxs):
            embs[idx_global, :] = emb_batch[i_local]
            processed[idx_global] = True

        embs.flush()
        np.save(proc_path, processed)

        if device.type == "cuda":
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        time.sleep(0.01)

    # write diagnostics
    with open(out_dir / "clap_audio_failed_loads.json", "w", encoding="utf-8") as f:
        json.dump(failed_loads, f, indent=2, ensure_ascii=False)
    with open(out_dir / "clap_audio_failed_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(failed_embeddings, f, indent=2, ensure_ascii=False)

    np.save(proc_path, processed)
    processed_count = int(processed.sum())

    print("[DONE]")
    print("Embeddings memmap:", embs_path)
    print("Processed flags:", proc_path)
    print("Resolved map:", resolved_csv)
    print("Missing guesses:", out_dir / "clap_audio_missing.json")
    print("Failed loads:", out_dir / "clap_audio_failed_loads.json")
    print("Failed embeds:", out_dir / "clap_audio_failed_embeddings.json")
    print(f"[SUMMARY] total rows: {n}, processed embeddings: {processed_count}, missing/unprocessed: {n-processed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV with PATH & TRACK_ID")
    parser.add_argument("--audio-root", required=True, help="Root folder containing audio subfolders (00..99)")
    parser.add_argument("--out-dir", default="./embs_audio", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    parser.add_argument("--clap-ckpt", type=str, default=None, help="CLAP checkpoint (.pt)")
    parser.add_argument("--clap-amodel", type=str, default="HTSAT-base", help="CLAP audio backbone")
    parser.add_argument("--sr", type=int, default=48000, help="SR for librosa reads")
    parser.add_argument("--no-trim", action="store_true", help="Do NOT try trimmed numeric fallback (int(numeric))")
    args = parser.parse_args()

    main(args.metadata, args.audio_root, args.out_dir,
         batch_size=args.batch_size,
         use_gpu=(not args.no_gpu),
         clap_ckpt=args.clap_ckpt,
         clap_amodel=args.clap_amodel,
         sr=args.sr,
         try_trim_numeric=(not args.no_trim))