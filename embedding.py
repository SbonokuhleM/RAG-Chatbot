# embedddings
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid #for generating unique ids
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from ingest import process_all_pdfs
from chunks import split_documents


class EmbeddingManager:
    """Handles document embedding generation"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the embedding model"""
        try:
            print(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print(
                f"Successfully loaded embedding model: "
                f"Generated {self.model.get_sentence_embedding_dimension()} dimensions"
            )

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""

        if self.model is None:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings


# initialize embedding manager
embedding_manager = EmbeddingManager()

        