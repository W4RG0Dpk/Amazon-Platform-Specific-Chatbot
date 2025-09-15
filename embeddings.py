# embeddings.py
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pathlib import Path

DEFAULT_MODEL_PATH = r"C:\amrita_uni\Projects\BeyondChats\model"

@lru_cache(maxsize=1)
def get_embedder(model_path: str = DEFAULT_MODEL_PATH) -> SentenceTransformer:
    """
    Return a cached SentenceTransformer instance loaded from model_path.
    """
    model_path = str(Path(model_path))
    print(f"[embeddings] Loading SentenceTransformer from: {model_path}")
    model = SentenceTransformer(model_path)
    return model
