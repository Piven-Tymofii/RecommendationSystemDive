import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import math
import logging
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_pipeline")

# optional dependencies imported lazily
try:
    import faiss
except Exception:
    faiss = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import laion_clap
    from laion_clap import CLAP_Module
except Exception:
    CLAP_Module = None
import torch

# numpy format helpers (used for header parsing)
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

# -----------------------
# helpers for reading .npy / raw float blobs
# -----------------------
def is_npy(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(6)
        return head == b"\x93NUMPY"
    except Exception:
        return False


def parse_npy_header(path: Path):
    """Return (data_offset, shape, dtype) for a .npy file."""
    with open(path, "rb") as f:
        major, minor = read_magic(f)
        if (major, minor) >= (2, 0):
            _hdr = read_array_header_2_0(f)
        else:
            _hdr = read_array_header_1_0(f)
        data_offset = f.tell()
    # Use np.load with mmap to safely get shape/dtype (no full allocation)
    arr = np.load(str(path), mmap_mode="r")
    shape = arr.shape
    dtype = arr.dtype
    # close memmap if present
    try:
        if hasattr(arr, "_mmap") and arr._mmap is not None:
            arr._mmap.close()
    except Exception:
        pass
    del arr
    return data_offset, shape, dtype


def read_rows_from_npy(path: Path, idxs: list, data_offset:int, D:int):
    """Read rows from a .npy file given the data offset and row dimension D.
       Returns np.array (len(idxs), D) dtype float32.
    """
    row_bytes = D * 4
    out = np.empty((len(idxs), D), dtype=np.float32)
    with open(path, "rb") as f:
        for i, idx in enumerate(idxs):
            offset = data_offset + int(idx) * row_bytes
            f.seek(offset)
            b = f.read(row_bytes)
            if len(b) != row_bytes:
                raise IOError(f"Read {len(b)} bytes, expected {row_bytes}; idx {idx}")
            out[i, :] = np.frombuffer(b, dtype=np.float32)
    return out


def make_row_reader(path: str, n_rows:int, dim:int):
    """Try memmap; if that fails (common on Windows), return a header-aware per-row reader.

    Returns: (get_subset_func, cleanup_func, (file_n_rows, file_dim))
    where get_subset_func(idxs) -> np.ndarray shape (len(idxs), D)
    """
    p = Path(path)
    # Try memmap first
    try:
        mm = np.memmap(str(p), dtype=np.float32, mode="r", shape=(n_rows, dim))
        logger.info("Using memmap for %s shape=(%d,%d)", p, n_rows, dim)
        def get_subset_memmap(idxs):
            return np.asarray(mm[list(idxs)], dtype=np.float32)
        def cleanup_memmap():
            try:
                del mm
            except Exception:
                pass
        return get_subset_memmap, cleanup_memmap, (n_rows, dim)
    except Exception as e:
        logger.warning("np.memmap failed (%s). Falling back to per-row reader.", e)

    # Fallback: parse header if npy
    if is_npy(p):
        try:
            data_offset, shape, dtype = parse_npy_header(p)
            if len(shape) == 2:
                header_n, header_d = shape
                if header_n == n_rows and header_d == dim:
                    logger.info("Header shape matches expected (%d,%d). Using per-row reader.", header_n, header_d)
                else:
                    logger.warning("Header shape %s differs from expected (%d,%d). Using header values.", shape, n_rows, dim)
                    n_rows, dim = int(header_n), int(header_d)
            else:
                logger.info("Parsed header shape %s", shape)
        except Exception as e:
            raise RuntimeError(f"Failed to parse .npy header: {e}")

        def get_subset_npy(idxs):
            return read_rows_from_npy(p, idxs, data_offset, dim)

        def cleanup_npy():
            return

        return get_subset_npy, cleanup_npy, (n_rows, dim)

    # If not npy, assume raw float32 blob and compute dim from file size
    sz = p.stat().st_size
    if sz % 4 != 0:
        raise RuntimeError("File size not multiple of 4 — cannot treat as float32 blob.")
    total_floats = sz // 4
    if total_floats % n_rows != 0:
        raise RuntimeError("Cannot infer D from raw blob; total_floats not divisible by n_rows.")
    inferred_d = total_floats // n_rows
    logger.info("Using raw float32 blob reader, inferred D=%d", inferred_d)
    def read_rows_raw(pathp, idxs, D=inferred_d):
        row_bytes = D * 4
        out = np.empty((len(idxs), D), dtype=np.float32)
        with open(pathp, "rb") as f:
            for i, idx in enumerate(idxs):
                offset = int(idx) * row_bytes
                f.seek(offset)
                b = f.read(row_bytes)
                if len(b) != row_bytes:
                    raise IOError(f"Read {len(b)} bytes, expected {row_bytes}; idx {idx}")
                out[i, :] = np.frombuffer(b, dtype=np.float32)
        return out
    return lambda idxs: read_rows_raw(str(p), idxs), lambda: None, (n_rows, inferred_d)


# -----------------------
# simple numeric helpers
# -----------------------
def l2_normalize_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = max(np.linalg.norm(x), eps)
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


# -----------------------
# main pipeline functions
# -----------------------
def build_shortlist_sbert(prompt, sbert_model, sb_embs_path, faiss_index_path, shortlist_k=300):
    emb = sbert_model.encode([prompt], convert_to_numpy=True)
    emb = emb.astype(np.float32)
    emb = l2_normalize_np(emb).reshape(1, -1)

    if faiss is not None and Path(faiss_index_path).exists():
        logger.info("Loading FAISS index %s", faiss_index_path)
        idx = faiss.read_index(str(faiss_index_path))
        D = idx.d
        if emb.shape[1] != D:
            logger.warning("SBERT dim mismatch (query %d vs index %d). Falling back to brute-force.", emb.shape[1], D)
        else:
            logger.info("Searching FAISS for top %d", shortlist_k)
            Dists, Idxs = idx.search(emb, shortlist_k)
            return Idxs[0].tolist()

    logger.info("FAISS not available or index mismatch — using brute force search over sbert_embs.npy")
    sb_embs = np.load(sb_embs_path, mmap_mode="r")
    sb_embs = l2_normalize_np(sb_embs)
    sims = (emb @ sb_embs.T).squeeze(0)
    topk = min(shortlist_k, sims.shape[0])
    idxs = np.argsort(-sims)[:topk].tolist()
    return idxs


def clap_rerank(prompt, clap, clap_audio_embs_path, candidate_idxs, n_rows=18486, dim=512, initial_batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    get_subset, cleanup, (file_n_rows, file_dim) = make_row_reader(clap_audio_embs_path, n_rows, dim)
    if (file_n_rows, file_dim) != (n_rows, dim):
        logger.warning("Using file dims (%d,%d) instead of requested (%d,%d).", file_n_rows, file_dim, n_rows, dim)
        n_rows, dim = file_n_rows, file_dim

    cand = [int(i) for i in candidate_idxs if 0 <= int(i) < n_rows]
    if not cand:
        cleanup()
        return np.array([], dtype=np.float32)

    with torch.no_grad():
        text_emb_raw = clap.get_text_embedding([prompt])
    text_emb = np.asarray(text_emb_raw, dtype=np.float32).reshape(1, -1)
    if text_emb.shape[1] != dim:
        logger.warning("CLAP dim mismatch text=%d audio=%d. Slicing/padding text to audio dim.", text_emb.shape[1], dim)
        if text_emb.shape[1] > dim:
            text_emb = text_emb[:, :dim]
        else:
            pad = np.zeros((1, dim - text_emb.shape[1]), dtype=np.float32)
            text_emb = np.concatenate([text_emb, pad], axis=1)
    text_emb = l2_normalize_np(text_emb)
    text_t = torch.from_numpy(text_emb).to(device)

    batch_size = initial_batch_size
    scores = np.zeros((len(cand),), dtype=np.float32)
    i = 0
    while i < len(cand):
        end = min(i + batch_size, len(cand))
        batch_idxs = cand[i:end]
        try:
            audio_batch = get_subset(batch_idxs)
            audio_batch = audio_batch.astype(np.float32)
            audio_batch = l2_normalize_np(audio_batch)
            a_t = torch.from_numpy(audio_batch).to(device)
            with torch.no_grad():
                s_t = (text_t @ a_t.T).squeeze(0).cpu().numpy()
            scores[i:end] = s_t.astype(np.float32)
            del a_t, s_t, audio_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            i = end
        except (OSError, MemoryError, RuntimeError) as e:
            logger.warning("Batch of size %d failed with %s — reducing batch size and retrying.", batch_size, e)
            if batch_size <= 1:
                logger.exception("Batch size 1 still fails; aborting CLAP rerank.")
                cleanup()
                raise
            batch_size = max(1, batch_size // 2)
            continue

    cleanup()
    return scores


def sbert_rerank(prompt, sbert_model, sb_embs_path, candidate_idxs):
    emb = sbert_model.encode([prompt], convert_to_numpy=True).astype(np.float32)
    emb = l2_normalize_np(emb).reshape(1,-1)
    sb_embs = np.load(sb_embs_path, mmap_mode="r")
    cand = [i for i in candidate_idxs if 0 <= i < sb_embs.shape[0]]
    if not cand:
        return []
    batch = sb_embs[cand].astype(np.float32)
    batch = l2_normalize_np(batch)
    sims = (emb @ batch.T).squeeze(0)
    results = [(ci, float(s)) for ci, s in zip(cand, sims.tolist())]
    results.sort(key=lambda x: -x[1])
    return results


def load_meta(meta_index_path):
    if not Path(meta_index_path).exists():
        logger.warning("meta_index.csv not found at %s", meta_index_path)
        return None
    meta = pd.read_csv(meta_index_path, dtype=str).fillna("")
    return meta

# -----------------------
# CLI run
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="User text prompt (quote it)")
    p.add_argument("--topk", type=int, default=5, help="How many final results to return")
    p.add_argument("--shortlist-k", type=int, default=300, help="How many candidates from FAISS to shortlist")
    p.add_argument("--sb_embs", default=r"..\db_create\data\embeddings\sbert_embs.npy", help="SBERT embeddings (n x d)")
    p.add_argument("--faiss", default=r"db_create\data\embeddings\sbert_text_index.faiss", help="FAISS index file")
    p.add_argument("--meta", default=r"..\db_create\data\embeddings\meta_index.csv", help="meta_index.csv path")
    p.add_argument("--clap-audio", default=r"..\db_create\data\embeddings\embs_audio\clap_audio_embs.npy", help="CLAP audio embeddings (n x d)")
    p.add_argument("--use-gpu", action="store_true", help="Try to run CLAP on CUDA (if available)")
    p.add_argument("--clap-amodel", default="HTSAT-base", help="CLAP amodel to init (if using laion_clap)")
    p.add_argument("--no-clap", action="store_true", help="Disable CLAP reranking and use SBERT-only rerank")
    args = p.parse_args()

    prompt = args.prompt
    logger.info("Prompt: %s", prompt)

    meta = load_meta(args.meta)
    if meta is None:
        logger.error("meta_index.csv not found. Aborting.")
        sys.exit(1)

    if SentenceTransformer is None:
        logger.error("sentence-transformers not installed. Install `sentence-transformers`.")
        sys.exit(1)
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    logger.info("SBERT model loaded (device=cpu)")

    candidate_idxs = build_shortlist_sbert(prompt, sbert, args.sb_embs, args.faiss, shortlist_k=args.shortlist_k)
    logger.info("Shortlist size: %d", len(candidate_idxs))

    final_scores = None
    if not args.no_clap and CLAP_Module is not None:
        device = torch.device("cuda:0") if (args.use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        logger.info("Initializing CLAP (amodel=%s) on device=%s", args.clap_amodel, device)
        clap = CLAP_Module(enable_fusion=False, device=device, amodel=args.clap_amodel)
        try:
            logger.info("Reranking with CLAP text->audio similarity.")
            scores = clap_rerank(prompt, clap, args.clap_audio, candidate_idxs)
            # pair candidates with scores and sort
            paired = list(zip(candidate_idxs, scores.tolist()))
            paired.sort(key=lambda x: -x[1])
            final_scores = paired[:args.topk]
        except Exception as e:
            logger.exception("CLAP rerank failed: %s. Falling back to SBERT rerank.", e)
            final_scores = sbert_rerank(prompt, sbert, args.sb_embs, candidate_idxs)[:args.topk]
    else:
        logger.info("CLAP unavailable or disabled; using SBERT-only rerank")
        final_scores = sbert_rerank(prompt, sbert, args.sb_embs, candidate_idxs)[:args.topk]

    if not final_scores:
        logger.info("No results.")
        return

    print("\nTop results:")
    rows = []
    for rank, (idx, score) in enumerate(final_scores, start=1):
        row = {}
        row["rank"] = rank
        row["index"] = int(idx)
        row["score"] = float(score)
        if "TRACK_ID" in meta.columns:
            row["TRACK_ID"] = meta.iloc[idx].get("TRACK_ID", "")
        if "caption" in meta.columns:
            row["caption"] = meta.iloc[idx].get("caption", "")
        if "PATH" in meta.columns:
            row["PATH"] = meta.iloc[idx].get("PATH", "")
        rows.append(row)
        print(f"{rank:2d}) idx={idx} score={score:.4f} track_id={row.get('TRACK_ID','')} caption={row.get('caption','')[:120]} path={row.get('PATH','')}")
    df_out = pd.DataFrame(rows)
    out_path = Path("last_recommendations.csv")
    df_out.to_csv(out_path, index=False)
    logger.info("Saved top results to %s", out_path)

if __name__ == "__main__":
    main()
