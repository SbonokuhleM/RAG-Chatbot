import os
import chromadb
import uuid
import numpy as np
from typing import List, Any

from chunks import split_documents, process_all_pdfs
from embedding import EmbeddingManager


class vectorStore:
    """Handles the vector store for document embeddings"""

    def __init__(
        self,
        collection_name: str = "document_embeddings",
        persist_directory: str = "data/db"
    ):

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self.initialize_vector_store()

    def initialize_vector_store(self):
        """Initialize the vector store"""

        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings",
                          "hnsw:space": "cosine"})

            print(
                f"Vector store initialized with collection: "
                f"{self.collection_name}"
            )

            print(
                f"Existing documents: {self.collection.count()}"
            )

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_document(
        self,
        documents: List[Any],
        embeddings: np.ndarray
    ):
        """Add documents and their embeddings to the vector store"""

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents"
            )

        print(
            f"Adding {len(documents)} documents to the vector store"
        )

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        # Prepare all data
        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            # Unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)

            # Text
            documents_text.append(doc.page_content)

            # Individual embedding
            embeddings_list.append(embedding.tolist())

        # Add everything to ChromaDB
        try:

            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )

            print(
                f"Successfully added {len(documents)} "
                f"documents to the vector store"
            )

        except Exception as e:

            print(
                f"Error adding documents to the vector store: {e}"
            )

            raise


# --------------------------------------------------
# Initialize vector store

vector_store = vectorStore()

# Load and split PDFs


dir_path = "data"

documents = process_all_pdfs(dir_path)

chunks = split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# Extract text
# --------------------------------------------------

texts = [doc.page_content for doc in chunks]


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

embedding_manager = EmbeddingManager()

embeddings = embedding_manager.generate_embeddings(texts)

print(f"Embedding shape: {embeddings.shape}")
# --------------------------------------------------
# Store embeddings in ChromaDB

vector_store.add_document(chunks,embeddings)
