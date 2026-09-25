'''
Semantic models will be implemented here. For a reference:
I have first integrated SBert (SemanticFeatureExtractor). It was good enough 92%.
Then I have integrated SciBert (which is better for scientific articles). The results were better 93.1%

RAM Optimization:
- Embeddings are computed batch-by-batch and saved to disk immediately.
- The full (N x dim) matrix is NEVER held in RAM simultaneously.
- Final result is assembled via np.memmap (disk-backed) → ~0 extra RAM.
- Only `batch_size` rows live in RAM at any given time during computation.
'''
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
from project_config import DATA_DIR
from transformers import logging as hf_logging # To silence transformers warnings

import torch # PyTorch is a deep learning framework
import scipy.sparse as sp # SciPy sparse matrices are used for efficient storage of sparse data
import numpy as np # NumPy is a library for numerical operations
import gc # Garbage collection, helps with memory management
import logging # To silence logging warnings
import warnings # To silence warning warnings

# Remove HuggingFace, httpx and unnecessary warnings
import os

os.environ["HF_HUB_VERBOSITY"] = "error" # Remove HugginFace Warning

logging.getLogger("httpx").setLevel(logging.WARNING)
hf_logging.set_verbosity_error()

class SemanticFeatureExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):  # small, fast, free
        self.model = SentenceTransformer(model_name)
        self.model_name = "minilm"
        self.embedding_dim = 384  # MiniLM output dimension

    def extract_embeddings(self, texts, filenames=None, desc="Extracting"):
        """
        Extract semantic embeddings using a pre-trained transformer model.
        RAM-efficient: streams results to disk, never holds full matrix in RAM.
        Returns a numpy memmap (disk-backed) array of shape (N, embedding_dim).
        """
        cache_dir = DATA_DIR / "features" / f"cache_{self.model_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if filenames is None:
            filenames = [f"doc_{i}" for i in range(len(texts))]

        n = len(texts)

        # 1. Find which files still need computing (validity check only, don't keep in RAM)
        texts_to_process = []
        indices_to_process = []

        for i, (text, fname) in enumerate(zip(texts, filenames)):
            cache_path = cache_dir / f"{fname}.npy"
            if cache_path.exists():
                try:
                    np.load(cache_path)  # quick validity check — discard immediately
                except (ValueError, OSError):
                    print(f"Warning: {fname}.npy looks like it is defect. It will be calculated again.")
                    cache_path.unlink()
                    texts_to_process.append(text)
                    indices_to_process.append(i)
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)

        # 2. Compute missing embeddings and save each to disk immediately
        if texts_to_process:
            print(f"\n[{desc}] Processing {len(texts_to_process)} NEW files "
                  f"(Found {n - len(texts_to_process)} in cache)")

            new_embeddings = self.model.encode(
                texts_to_process,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32,
            )

            for idx, text_idx in enumerate(indices_to_process):
                emb = new_embeddings[idx]
                fname = filenames[text_idx]
                np.save(cache_dir / f"{fname}.npy", emb)

            # Free the big array immediately after saving (don't carry it forward)
            del new_embeddings
            gc.collect()
        else:
            print(f"\n[{desc}] All {n} files loaded from cache!")

        # 3. Assemble result via disk-backed memmap (0 extra RAM)
        memmap_path = cache_dir / f"_combined_{desc.replace(' ', '_')}.dat"
        result = np.memmap(
            memmap_path, dtype="float32", mode="w+", shape=(n, self.embedding_dim)
        )
        for i, fname in enumerate(tqdm(filenames, desc=f"[{desc}] Loading from disk")):
            result[i] = np.load(cache_dir / f"{fname}.npy")

        result.flush()
        # Re-open read-only so downstream code can't accidentally mutate it
        result = np.memmap(memmap_path, dtype="float32", mode="r", shape=(n, self.embedding_dim))
        return result

    def combine_with_tfidf(self, tfidf_features, semantic_features):
        """TF-IDF + Semantic concatenate efficiently"""
        semantic_sparse = sp.csr_matrix(semantic_features)
        combined = sp.hstack([tfidf_features, semantic_sparse])
        return combined


class SciBERTSemanticFeatureExtractor:
    def __init__(
            self,
            model_name="allenai/scibert_scivocab_uncased",
            device=None,
            batch_size=4,
    ):
        # Device settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SciBERT] Using device: {self.device}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = "scibert"
        self.embedding_dim = 768  # SciBERT CLS-token output dimension
        self.batch_size = batch_size

    def extract_embeddings(self, texts, filenames=None, batch_size=None, desc="Extracting Embeddings"):
        """
        Extract SciBERT embeddings using CLS token.

        RAM strategy:
          - Each batch is computed and immediately saved to its .npy cache file.
          - After all batches, the result is assembled via np.memmap.
            The full (N x 768) matrix is NEVER held in RAM.
            Only `batch_size` rows live in RAM at a time.
        Returns: numpy memmap array of shape (N, 768).
        """
        bs = batch_size or self.batch_size  # allow per-call override

        if isinstance(texts, str):
            texts = [texts]
            filenames = ["single_doc"] if filenames is None else filenames

        if filenames is None:
            filenames = [f"doc_{i}" for i in range(len(texts))]

        cache_dir = DATA_DIR / "features" / f"cache_{self.model_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        n = len(texts)

        # 1. Find which files still need computing
        texts_to_process = []
        indices_to_process = []

        for i, (text, fname) in enumerate(zip(texts, filenames)):
            cache_path = cache_dir / f"{fname}.npy"
            if cache_path.exists():
                try:
                    np.load(cache_path)  # validity check only — discard immediately
                except (ValueError, OSError):
                    print(f"Warning: {fname}.npy looks like it is defect. It will be calculated again")
                    cache_path.unlink()
                    texts_to_process.append(text)
                    indices_to_process.append(i)
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)

        # 2. Compute missing embeddings batch-by-batch, save each batch to disk immediately
        if texts_to_process:
            print(f"\n[{desc}] Processing {len(texts_to_process)} NEW files "
                  f"(Found {n - len(texts_to_process)} in cache)")

            self.model.eval()

            with torch.no_grad():
                for b_start in tqdm(range(0, len(texts_to_process), bs), desc=desc):
                    batch_texts   = texts_to_process[b_start: b_start + bs]
                    batch_indices = indices_to_process[b_start: b_start + bs]

                    # Handle fully-empty texts
                    if all(len(t.strip()) == 0 for t in batch_texts):
                        for orig_idx in batch_indices:
                            fname = filenames[orig_idx]
                            np.save(cache_dir / f"{fname}.npy",
                                    np.zeros(self.embedding_dim, dtype="float32"))
                        continue

                    inputs = self.tokenizer(
                        batch_texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt"
                    ).to(self.device)

                    try:
                        outputs = self.model(**inputs)
                        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    except torch.cuda.OutOfMemoryError:
                        print(f"CUDA OOM falling back to CPU for this batch.")
                        torch.cuda.empty_cache()
                        self.model.to("cpu")
                        inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                        self.model.to(self.device)

                    # Save each embedding immediately → free RAM
                    for emb_idx, orig_idx in enumerate(batch_indices):
                        fname = filenames[orig_idx]
                        np.save(cache_dir / f"{fname}.npy",
                                cls_embeddings[emb_idx].astype("float32"))

                    # Explicit cleanup every batch
                    del inputs, outputs, cls_embeddings
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
        else:
            print(f"\n[{desc}] All {n} files loaded from cache!")

        # 3. Assemble result via disk-backed memmap (0 extra RAM)
        memmap_path = cache_dir / f"_combined_{desc.replace(' ', '_')}.dat"
        result = np.memmap(
            memmap_path, dtype="float32", mode="w+", shape=(n, self.embedding_dim)
        )
        for i, fname in enumerate(tqdm(filenames, desc=f"[{desc}] Loading from disk")):
            result[i] = np.load(cache_dir / f"{fname}.npy")

        result.flush()
        result = np.memmap(memmap_path, dtype="float32", mode="r", shape=(n, self.embedding_dim))
        return result

    def combine_with_tfidf(self, tfidf_features, semantic_features):
        """TF-IDF + SciBERT embeddings concatenation"""
        semantic_sparse = sp.csr_matrix(semantic_features)
        combined = sp.hstack([tfidf_features, semantic_sparse])
        return combined
