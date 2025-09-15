# retrieval.py
import faiss
import pickle
import numpy as np
from pathlib import Path
from embeddings import get_embedder

STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "faiss.index"
META_PATH = STORAGE_DIR / "meta.pkl"

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.20

# Load index & metadata at import time
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError("Run ingest.py first to create storage/faiss.index and storage/meta.pkl")

_index = faiss.read_index(str(INDEX_PATH))
with open(str(META_PATH), "rb") as f:
    _metadata = pickle.load(f)

_embedder = get_embedder()

def search(query: str, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_THRESHOLD):
    """
    Returns: (retrieved_list, is_ood, max_score)
     - retrieved_list: list of {idx,line_no,text,score}
     - is_ood: True when max_score < threshold
    """
    q_emb = _embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = _index.search(q_emb, top_k)
    scores = D[0].tolist()
    ids = I[0].tolist()
    retrieved = []
    for score, doc_idx in zip(scores, ids):
        if doc_idx < 0:
            continue
        meta = _metadata[doc_idx]
        retrieved.append({"idx": int(doc_idx), "line_no": int(meta["line_no"]), "text": meta["text"], "score": float(score)})
    max_score = max(scores) if scores else 0.0
    is_ood = max_score < threshold
    # Debug log (printed to uvicorn console)
    print(f"[retrieval] query={query!r} top_k={top_k} max_score={max_score:.4f} is_ood={is_ood}")
    return retrieved, is_ood, float(max_score)
