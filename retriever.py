from typing import Any

from db import vectorStore
from embedding import EmbeddingManager


class RAGretriever:
    """Handles query-based retrieval of relevant documents from the vector store"""

    def __init__(
        self,
        vector_store: vectorStore,
        embedding_manager: EmbeddingManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents based on the query"""

        print(f"Retrieving documents for query: {query}")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")

        retrieved_docs = []

        try:
            # Generate embedding for the query
            query_embedding = self.embedding_manager.generate_embeddings(
                [query]
            )[0]

            # Query the vector store
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            if results["documents"] and results["documents"][0]:

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (
                    doc_id,
                    document,
                    metadata,
                    distance
                ) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):

                    # Convert distance to a simple similarity score
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:

                        retrieved_docs.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(
                    f"Retrieved {len(retrieved_docs)} documents "
                    f"after filtering by score threshold"
                )

            else:
                print("No documents retrieved from the vector store")

        except Exception as e:
            print(f"Error retrieving documents: {e}")

        return retrieved_docs



rag_retriever=RAGretriever(vector_store=vector_store, embedding_manager=embedding_manager)

rag_retriever.retrieve("What are disadvantages of LSTMs")