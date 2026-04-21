# Semantic Models will be integrated here

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
from project_config import DATA_DIR
import torch
import scipy.sparse as sp
import numpy as np


class SemanticFeatureExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'): # small, fast, free
        self.model = SentenceTransformer(model_name)
        self.model_name = "minilm"

    def extract_embeddings(self, texts, filenames=None, desc="Extracting"):
        """
        Extract semantic embeddings using a pre-trained transformer model
        Semantic vectors capture contextual meaning of texts.
        """
        # Prepare cache file
        cache_dir = DATA_DIR / "features" / f"cache_{self.model_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if filenames is None:
            filenames = [f"doc_{i}" for i in range(len(texts))]

        final_embeddings = [None] * len(texts)
        texts_to_process = []
        indices_to_process = []

        for i, (text, fname) in enumerate(zip(texts, filenames)):
            cache_path = cache_dir / f"{fname}.npy"
            if cache_path.exists():
                final_embeddings[i] = np.load(cache_path)
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        # Calculate only the missing articles 
        if len(texts_to_process) > 0:
            print(f"\n[{desc}] Processing {len(texts_to_process)} NEW files (Found {len(texts) - len(texts_to_process)} in cache)")
            
            # SentenceTransformer encode
            new_embeddings = self.model.encode(texts_to_process, convert_to_numpy=True, show_progress_bar=True)
            
            # Save new files into the disk
            for idx, text_idx in enumerate(indices_to_process):
                emb = new_embeddings[idx]
                fname = filenames[text_idx]
                np.save(cache_dir / f"{fname}.npy", emb)
                final_embeddings[text_idx] = emb
        else:
            print(f"\n[{desc}] All {len(texts)} files loaded from cache!")

        return np.vstack(final_embeddings)

    
    def combine_with_tfidf(self, tfidf_features, semantic_features):
        """TF-IDF + Semantic concatenate efficiently"""
        semantic_sparse = sp.csr_matrix(semantic_features)

        # Horizontally stack the sparse matrices
        combined = sp.hstack([tfidf_features, semantic_sparse])
        return combined

class SciBERTSemanticFeatureExtractor:
    def __init__(
            self, 
            model_name="allenai/scibert_scivocab_uncased",
            device=None
        ):
            # Device settings
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_name = "scibert"
    
    def extract_embeddings(self, texts, filenames=None, batch_size=4, desc = "Extracting Embeddings"): # Batch size 32 tired the CPU
        """
        Extract SciBERT embeddings using CLS token
        Returns:numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
            filenames = ["single_doc"] if filenames is None else filenames

        if filenames is None:
            filenames = [f"doc_{i}" for i in range(len(texts))]

        # Prepare cache file
        cache_dir = DATA_DIR / "features" / f"cache_{self.model_name}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        final_embeddings = [None] * len(texts)
        texts_to_process = []
        indices_to_process = []

        # Controll which files exist in cache
        for i,(text, fname) in enumerate(zip(texts, filenames)):
            cache_path = cache_dir / f"{fname}.npy"
            if cache_path.exists():
                final_embeddings[i] = np.load(cache_path)
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        # if a file doesn't exist
        if len(texts_to_process) > 0:
            print(f"\n[{desc}] Processing {len(texts_to_process)} NEW files (Found {len(texts) - len(texts_to_process)} in cache)")
            self.model.eval()
            new_embeddings = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts_to_process), batch_size), desc=desc):
                    batch_texts = texts_to_process[i:i + batch_size]
                    # Bu grubun orijinal indexleri (hangi dosyaya ait oldukları)
                    batch_indices = indices_to_process[i:i + batch_size]

                    inputs = self.tokenizer(
                        batch_texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt"
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    # 
                    for emb_idx, emb in enumerate(cls_embeddings):
                        original_idx = batch_indices[emb_idx]
                        fname = filenames[original_idx]
                        
                        # Type directly to disk
                        np.save(cache_dir / f"{fname}.npy", emb)
                    
                        final_embeddings[original_idx] = emb
        else:
            print(f"\n[{desc}] All {len(texts)} files loaded from cache!")

        # numpy + vstack 
        return np.vstack(final_embeddings)

    def combine_with_tfidf(self, tfidf_features, semantic_features):
        """
        TF-IDF + SciBERT embeddings concatenation
        """
        semantic_sparse = sp.csr_matrix(semantic_features)
        combined = sp.hstack([tfidf_features, semantic_sparse])
        return combined
