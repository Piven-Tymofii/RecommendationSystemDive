"""
Workflow:
  1) Encode user prompt with SBERT -> search FAISS index -> get top-N shortlist (default 300)
  2) Encode user prompt with CLAP text encoder -> cosine similarity with precomputed CLAP audio embeddings of the shortlist
  3) Return final top-k (default 5)

Note: Change paths via CLI args.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import math
import logging

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

# -----------------------
# helpers
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
# main pipeline
# -----------------------
def build_shortlist_sbert(prompt, sbert_model, sb_embs_path, faiss_index_path, shortlist_k=300):
    """
    Return list of candidate indices (length <= shortlist_k)
    """
    # 1) encode prompt
    emb = sbert_model.encode([prompt], convert_to_numpy=True)
    emb = emb.astype(np.float32)
    emb = l2_normalize_np(emb).reshape(1, -1)
    scores_sbert = None
    # 2) try faiss
    if faiss is not None and Path(faiss_index_path).exists():
        logger.info("Loading FAISS index %s", faiss_index_path)
        idx = faiss.read_index(str(faiss_index_path))
        # ensure index dimension matches
        D = idx.d
        if emb.shape[1] != D:
            logger.warning("SBERT dim mismatch (query %d vs index %d). Falling back to brute-force.", emb.shape[1], D)
        else:
            logger.info("Searching FAISS for top %d", shortlist_k)
            # inner product on normalized vectors ~ cosine
            Dists, Idxs = idx.search(emb, shortlist_k)
            scores_sbert = 1 - Dists[0]  # convert distance to similarity score in sbert to compare with final results
            return Idxs[0].tolist(), scores_sbert.tolist()
    # fallback to brute force using saved sbert_embs.npy
    logger.info("FAISS not available or index mismatch — using brute force search over sbert_embs.npy")
    sb_embs = np.load(sb_embs_path, mmap_mode="r")
    # normalize just in case
    sb_embs = l2_normalize_np(sb_embs)
    sims = (emb @ sb_embs.T).squeeze(0)  # shape (N,)
    topk = min(shortlist_k, sims.shape[0])
    idxs = np.argsort(-sims)[:topk].tolist()
    scores_sbert = sims[idxs].tolist()
    return idxs, scores_sbert

def clap_rerank(prompt, clap_module, clap_audio_embs_path, candidate_idxs, batch_size=256, device=None):
    """
    Compute CLAP text embedding for prompt and dot against audio embeddings for candidates.
    Returns list of (idx, score) sorted by score desc.
    """
    # load just candidates' audio emb rows
    # to avoid loading whole huge matrix when it is giant, we'll load memmap and index rows
    audio_embs = np.load(clap_audio_embs_path, mmap_mode="r", allow_pickle=True)
    # make sure candidate indices are in-bounds
    cand = [i for i in candidate_idxs if 0 <= i < audio_embs.shape[0]]
    if len(cand) == 0:
        return []

    # get CLAP text embedding for prompt
    with torch.no_grad():
        text_emb_raw = clap_module.get_text_embedding([prompt])
    text_emb = to_numpy(text_emb_raw).astype(np.float32).reshape(1, -1)
    text_emb = l2_normalize_np(text_emb)

    # compute similarity in batches if many candidates
    scores = []
    for i in range(0, len(cand), batch_size):
        batch_idxs = cand[i:i+batch_size]
        batch_emb = audio_embs[batch_idxs]  # memmap slice
        # ensure float32 and normalized
        batch_emb = batch_emb.astype(np.float32)
        batch_emb = l2_normalize_np(batch_emb)
        s = (text_emb @ batch_emb.T).squeeze(0)  # shape (B,)
        for idx_local, sc in zip(batch_idxs, s.tolist()):
            scores.append((idx_local, float(sc)))
    # sort by score desc
    scores.sort(key=lambda x: -x[1])
    return scores

def sbert_rerank(prompt, sbert_model, sb_embs_path, candidate_idxs):
    """
    Fallback rerank: use SBERT similarity between prompt and candidate combined_text SBERT embeddings.
    This requires sb_embs.npy to align with meta rows.
    """
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
    p.add_argument("--prompt", required=True, help="User text prompt (quote it) or pass --prompt-file")
    p.add_argument("--topk", type=int, default=5, help="How many final results to return")
    p.add_argument("--shortlist-k", type=int, default=300, help="How many candidates from FAISS to shortlist")
    p.add_argument("--sb_embs", default=r"..\Main_DataSet_Creation\data\embeddings\sbert_embs.npy", help="SBERT embeddings (n x d)")
    p.add_argument("--faiss", default=r"Main_DataSet_Creation\data\embeddings\sbert_text_index.faiss", help="FAISS index file")
    p.add_argument("--meta", default=r"..\Main_DataSet_Creation\data\embeddings\meta_index.csv", help="meta_index.csv path")
    p.add_argument("--clap-audio", default=r"..\Main_DataSet_Creation\data\embeddings\embs_audio\clap_audio_processed.npy", help="CLAP audio embeddings (n x d)")
    p.add_argument("--use-gpu", action="store_true", help="Try to run CLAP on CUDA (if available)")
    p.add_argument("--clap-amodel", default="HTSAT-base", help="CLAP amodel to init (if using laion_clap)")
    p.add_argument("--no-clap", action="store_true", help="Disable CLAP reranking and use SBERT-only rerank")
    args = p.parse_args()


    prompt = args.prompt
    logger.info("Prompt: %s", prompt)

    # load meta
    meta = load_meta(args.meta)
    if meta is None:
        logger.error("meta_index.csv not found. Aborting.")
        sys.exit(1)

    # 1) SBERT encode prompt and shortlist with FAISS (or brute force)
    if SentenceTransformer is None:
        logger.error("sentence-transformers not installed. Install `sentence-transformers`.")
        sys.exit(1)
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    logger.info("SBERT model loaded (device=cpu)")

    candidate_idxs, sbert_scores = build_shortlist_sbert(prompt, sbert, args.sb_embs, args.faiss, shortlist_k=args.shortlist_k)
    logger.info("Shortlist size: %d", len(candidate_idxs))

    # 2) CLAP rerank (preferred)
    final_scores = None
    if not args.no_clap and CLAP_Module is not None:
        # init CLAP
        device = torch.device("cuda:0") if (args.use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        logger.info("Initializing CLAP (amodel=%s) on device=%s", args.clap_amodel, device)
        clap = CLAP_Module(enable_fusion=False, device=device, amodel=args.clap_amodel)
        # attempt to compute and rerank
        try:
            logger.info("Reranking with CLAP text->audio similarity.")
            scores = clap_rerank(prompt, clap, args.clap_audio, candidate_idxs)
            final_scores = scores[:args.topk]
        except Exception as e:
            logger.exception("CLAP rerank failed: %s. Falling back to SBERT rerank.", e)
            final_scores = sbert_rerank(prompt, sbert, args.sb_embs, candidate_idxs)[:args.topk]
    else:
        logger.info("CLAP unavailable or disabled; using SBERT-only rerank")
        final_scores = sbert_rerank(prompt, sbert, args.sb_embs, candidate_idxs)[:args.topk]

    # Present results
    if not final_scores:
        logger.info("No results.")
        return

    print("\nTop results:")
    results = []

    # Create dataframe for easier mapping
    meta_df = meta.copy()

    # store both SBERT and CLAP scores
    for rank, idx in enumerate(candidate_idxs[:len(scores)], 1):
        sbert_score = float(sbert_scores[idx]) if 'sbert_scores' in locals() else None
        clap_score = float(scores[idx]) if scores is not None else None
        delta = (clap_score - sbert_score) if (sbert_score is not None and clap_score is not None) else None

        row = meta_df.iloc[idx]
        results.append({
            "rank": rank,
            "idx": idx,
            "track_id": row.get("TRACK_ID", ""),
            "caption": row.get("caption", "")[:150],
            "path": row.get("PATH", ""),
            "SBERT_score": round(sbert_score, 4) if sbert_score else None,
            "CLAP_score": round(clap_score, 4) if clap_score else None,
            "Score_change": round(delta, 4) if delta else None,
        })

    # sort by CLAP_score descending (if available)
    if any(r["CLAP_score"] is not None for r in results):
        results = sorted(results, key=lambda x: x["CLAP_score"] or 0, reverse=True)

    out_df = pd.DataFrame(results)
    out_path = "last_recommendations.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved detailed comparison results to {out_path}")

if __name__ == "__main__":
    main()
