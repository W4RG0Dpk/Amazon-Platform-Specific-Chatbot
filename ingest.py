# ingest.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from embeddings import get_embedder

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = STORAGE_DIR / "faiss.index"
META_PATH = STORAGE_DIR / "meta.pkl"
EMB_VEC_PATH = STORAGE_DIR / "embeddings.npy"

def ingest_lines(lines, model_path: str = None):
    """
    Ingest lines (list of strings) into FAISS IndexFlatIP after embedding + L2 normalization.
    Writes index, embeddings and metadata to storage.
    """
    model = get_embedder(model_path) if model_path else get_embedder()
    texts = [ln.strip() for ln in lines if ln and ln.strip()]
    if not texts:
        raise ValueError("No lines provided for ingestion")

    print(f"[ingest] Encoding {len(texts)} lines ...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    # normalize for cosine search using inner product
    faiss.normalize_L2(embeddings.astype("float32"))
    dim = embeddings.shape[1]
    print(f"[ingest] building IndexFlatIP with dim={dim} ...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(EMB_VEC_PATH), embeddings)
    metadata = [{"line_no": i+1, "text": t} for i, t in enumerate(texts)]
    with open(str(META_PATH), "wb") as f:
        pickle.dump(metadata, f)
    print(f"[ingest] saved index -> {INDEX_PATH}, meta -> {META_PATH}, embeddings -> {EMB_VEC_PATH}")
    return True

def ingest_file(path="amazon_help_doc.txt", model_path: str = None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    with p.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return ingest_lines(lines, model_path=model_path)

if __name__ == "__main__":
    ingest_file()
