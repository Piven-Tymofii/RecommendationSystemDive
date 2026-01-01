#!/usr/bin/env python3
"""
Build and store text embeddings:
 - SBERT embeddings (for FAISS shortlist)
 - CLAP text embeddings (to compare with CLAP audio embeddings)

Saves:
 - sbert_embs.npy         (float32, shape N x d_sbert)  (normalized)
 - clap_text_embs.npy     (float32, shape N x d_clap)   (normalized)
 - meta_index.csv         (cols: index, TRACK_ID, caption, combined_text, Mood/Theme_TAG, Genre_TAG, Instrument_TAG)
 - sbert_text_index.faiss (optional FAISS index for SBERT embeddings)

Usage example:
 python build_text_embeddings.py \
   --metadata path/to/meta.csv \
   --captions path/to/captions.csv \
   --out-dir ./embs \
   --clap-ckpt ../model_checkpoint/music_audioset_epoch_15_esc_90.14.pt \
   --clap-amodel HTSAT-base
"""
import os
import argparse
import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import gc

# sentence-transformers and CLAP
from sentence_transformers import SentenceTransformer

try:
    import laion_clap
    from laion_clap import CLAP_Module
except Exception:
    CLAP_Module = None

# FAISS (optional)
try:
    import faiss
except Exception:
    faiss = None

# -------------------------
# Helpers
# -------------------------
def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x) + eps
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def to_numpy(x):
    """Return numpy array for torch tensor or numpy-like."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def normalize_tag_field(s: str) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return " ".join([str(t).strip() for t in parsed if t])
        except Exception:
            pass
    return s

def build_combined_text(row, use_tags=True):
    caption = (row.get("caption") or "").strip()
    mood = normalize_tag_field(row.get("Mood/Theme_TAG", ""))
    genre = normalize_tag_field(row.get("Genre_TAG", ""))
    instr = normalize_tag_field(row.get("Instrument_TAG", ""))
    tags = " ".join(filter(None, [mood, genre, instr]))
    if use_tags and tags:
        combined = f"{caption} | tags: {tags}"
    else:
        combined = caption
    return combined

# -------------------------
# Main logic
# -------------------------
def main(metadata_csv: str, captions_csv: str, out_dir: str,
         batch_size_sbert: int = 128, batch_size_clap: int = 64,
         use_gpu_clap: bool = True, clap_ckpt: str = None, clap_amodel: str = "HTSAT-base"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading CSVs...")
    meta = pd.read_csv(metadata_csv, dtype=str).fillna("")
    caps = pd.read_csv(captions_csv, dtype=str).fillna("")
    id_col = "TRACK_ID"
    df = pd.merge(meta, caps, on=id_col, how="left", suffixes=("","_cap"))
    # ensure caption column present
    if "caption" not in df.columns:
        for c in df.columns:
            if c.lower().strip() == "caption":
                df = df.rename(columns={c: "caption"})
                break
    df["caption"] = df["caption"].fillna("")
    print(f"[INFO] Loaded {len(df)} rows.")

    # combined text
    print("[INFO] Building combined_text field...")
    df["combined_text"] = df.apply(build_combined_text, axis=1)
    meta_map_path = out_dir / "meta_index.csv"
    df[[id_col, "caption", "combined_text", "Mood/Theme_TAG", "Genre_TAG", "Instrument_TAG"]].reset_index().to_csv(meta_map_path, index=False)
    print(f"[INFO] Saved meta index to {meta_map_path}")

    # ---------------------------
    # SBERT embeddings
    # ---------------------------
    print("[INFO] Loading SBERT model (all-MiniLM-L6-v2)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    texts = df["combined_text"].tolist()
    n = len(texts)
    emb_dim_sbert = sbert_model.get_sentence_embedding_dimension()
    print(f"[INFO] SBERT embed dim = {emb_dim_sbert}, device = {device}")

    sb_embs = np.zeros((n, emb_dim_sbert), dtype=np.float32)
    print("[INFO] Computing SBERT embeddings in batches...")
    for i in tqdm(range(0, n, batch_size_sbert), desc="sbert"):
        batch_texts = texts[i:i+batch_size_sbert]
        emb = sbert_model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype(np.float32)
        sb_embs[i:i+len(emb)] = emb

    # normalize and save
    sb_embs = l2_normalize_np(sb_embs)
    sb_out = out_dir / "sbert_embs.npy"
    np.save(sb_out, sb_embs)
    print(f"[INFO] Saved SBERT embeddings -> {sb_out} (shape={sb_embs.shape})")

    # build FAISS index if available
    if faiss is not None:
        print("[INFO] Building FAISS Index (IndexFlatIP) for normalized SBERT embeddings ...")
        d = sb_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(sb_embs)
        faiss_out = out_dir / "sbert_text_index.faiss"
        faiss.write_index(index, str(faiss_out))
        print(f"[INFO] FAISS index saved -> {faiss_out}")
    else:
        print("[WARN] faiss not available. Install faiss-cpu (pip install faiss-cpu) if you want a FAISS index.")

    # ---------------------------
    # CLAP text embeddings
    # ---------------------------
    if CLAP_Module is None:
        print("[WARN] laion_clap not available. Skipping CLAP text embeddings. Install laion_clap and the checkpoint.")
        return

    clap_device = torch.device("cuda:0") if (torch.cuda.is_available() and use_gpu_clap) else torch.device("cpu")
    print(f"[INFO] Initializing CLAP model on {clap_device} with amodel={clap_amodel} ...")
    clap = CLAP_Module(enable_fusion=False, device=clap_device, amodel=clap_amodel)

    if clap_ckpt:
        print("[INFO] Trying to load CLAP checkpoint:", clap_ckpt)
        try:
            clap.load_ckpt(ckpt=clap_ckpt)
            print("[INFO] CLAP checkpoint loaded successfully.")
        except Exception as e:
            print("[ERROR] clap.load_ckpt() raised an exception:")
            print(str(e))
            print("\n[HELP] Checkpoint/backbone mismatch is common. Try another --clap-amodel value (e.g. 'HTSAT-base', 'HTS-xxx', or the model name the checkpoint expects).")
            print("You can also try loading without ckpt and then calling clap.load_ckpt() later with a different amodel.")
            raise

    # compute CLAP text embeddings in batches
    print("[INFO] Computing CLAP text embeddings in batches...")
    clap_embs_list = []
    for i in tqdm(range(0, n, batch_size_clap), desc="clap_text"):
        batch_texts = df["combined_text"].iloc[i:i+batch_size_clap].tolist()
        with torch.no_grad():
            out = clap.get_text_embedding(batch_texts)
        arr = to_numpy(out).astype(np.float32)
        clap_embs_list.append(arr)

        # free memory aggressively if using GPU
        if clap_device.type == "cuda":
            try:
                torch.cuda.synchronize(clap_device)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    clap_embs = np.vstack(clap_embs_list)
    clap_embs = l2_normalize_np(clap_embs)
    clap_out = out_dir / "clap_text_embs.npy"
    np.save(clap_out, clap_embs)
    print(f"[INFO] Saved CLAP text embeddings -> {clap_out} (shape={clap_embs.shape})")

    print("[DONE] Artifacts saved in:", str(out_dir))

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV (must include TRACK_ID)")
    parser.add_argument("--captions", required=True, help="Path to captions CSV (TRACK_ID, caption)")
    parser.add_argument("--out-dir", default="./embs", help="Directory to save embeddings and index")
    parser.add_argument("--batch-sbert", type=int, default=128, help="SBERT batch size")
    parser.add_argument("--batch-clap", type=int, default=64, help="CLAP batch size")
    parser.add_argument("--no-gpu-clap", action="store_true", help="Do not attempt to run CLAP on GPU")
    parser.add_argument("--clap-ckpt", type=str, default=None, help="Path to CLAP checkpoint (.pt)")
    parser.add_argument("--clap-amodel", type=str, default="HTSAT-base", help="CLAP audio backbone variant ( e.g. 'HTSAT-base' )")
    args = parser.parse_args()

    main(args.metadata, args.captions, args.out_dir,
         batch_size_sbert=args.batch_sbert,
         batch_size_clap=args.batch_clap,
         use_gpu_clap=(not args.no_gpu_clap),
         clap_ckpt=args.clap_ckpt,
         clap_amodel=args.clap_amodel)
