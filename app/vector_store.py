import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from google import genai
from app.config import settings

class VectorStore:
    """
    Lightweight, persistent Vector Store using Google Gemini Embeddings
    and NumPy cosine similarity search.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or settings.VECTOR_DB_FILE
        self.documents: List[Dict[str, Any]] = []
        self._vectors: Optional[np.ndarray] = None
        self.load()

    def _get_client(self) -> genai.Client:
        """Initialize Google GenAI client."""
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add your Gemini API key in the .env file."
            )
        return genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a list of texts using Gemini API.
        Processes in batches to avoid payload limits.
        """
        client = self._get_client()
        embeddings: List[List[float]] = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=batch,
            )
            for emb in response.embeddings:
                embeddings.append(emb.values)

        return embeddings

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generates embedding for a single user query."""
        client = self._get_client()
        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=query,
        )
        # response.embeddings can have 1 item
        if response.embeddings:
            return response.embeddings[0].values
        raise RuntimeError("Failed to generate embedding for the query.")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds and stores document chunks.
        """
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.generate_embeddings(texts)

        for chunk, vector in zip(chunks, vectors):
            record = {
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "embedding": vector
            }
            self.documents.append(record)

        self._refresh_vectors()
        self.save()
        return len(chunks)

    def _refresh_vectors(self):
        """Constructs a NumPy matrix for vectorized similarity computation."""
        if self.documents:
            self._vectors = np.array([doc["embedding"] for doc in self.documents], dtype=np.float32)
        else:
            self._vectors = None

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant chunks using cosine similarity.
        """
        if not self.documents or self._vectors is None:
            return []

        top_k = top_k or settings.TOP_K_RESULTS
        query_vector = np.array(self.generate_query_embedding(query), dtype=np.float32)

        # Filter candidate indices if document_filter is provided
        if document_filter:
            candidate_indices = [
                i for i, doc in enumerate(self.documents)
                if doc["document_name"].lower() == document_filter.lower()
            ]
            if not candidate_indices:
                return []
            candidate_vectors = self._vectors[candidate_indices]
            docs_subset = [self.documents[i] for i in candidate_indices]
        else:
            candidate_vectors = self._vectors
            docs_subset = self.documents

        # Cosine similarity: (A . B) / (||A|| * ||B||)
        norm_candidates = np.linalg.norm(candidate_vectors, axis=1)
        norm_query = np.linalg.norm(query_vector)

        # Prevent division by zero
        zero_mask = (norm_candidates == 0) | (norm_query == 0)
        norm_product = norm_candidates * norm_query
        norm_product[zero_mask] = 1e-10

        similarities = np.dot(candidate_vectors, query_vector) / norm_product
        similarities[zero_mask] = 0.0

        # Sort indices by descending score
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = docs_subset[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "document_name": doc["document_name"],
                "page_number": doc["page_number"],
                "text": doc["text"],
                "similarity_score": float(similarities[idx])
            })

        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns summary of all unique indexed documents."""
        docs_summary: Dict[str, Dict[str, Any]] = {}
        for doc in self.documents:
            name = doc["document_name"]
            if name not in docs_summary:
                docs_summary[name] = {
                    "document_name": name,
                    "total_chunks": 0,
                    "pages": set()
                }
            docs_summary[name]["total_chunks"] += 1
            docs_summary[name]["pages"].add(doc["page_number"])

        return [
            {
                "document_name": name,
                "total_chunks": info["total_chunks"],
                "total_pages": len(info["pages"])
            }
            for name, info in docs_summary.items()
        ]

    def clear(self):
        """Clears all stored documents and vectors."""
        self.documents = []
        self._vectors = None
        if self.storage_path.exists():
            self.storage_path.unlink()

    def save(self):
        """Persists the vector store to disk in JSON format."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def load(self):
        """Loads the vector store from disk if present."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self._refresh_vectors()
            except Exception:
                self.documents = []
                self._vectors = None
