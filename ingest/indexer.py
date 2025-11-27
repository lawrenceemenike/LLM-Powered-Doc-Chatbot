import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os

class ChromaIndexer:
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "documents"):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Indexes documents into ChromaDB.
        
        Args:
            documents: List of document dictionaries (must contain 'chunk_id', 'text', and metadata).
            embeddings: List of embedding vectors corresponding to the documents.
        """
        ids = [doc["chunk_id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k not in ["text", "chunk_id"]} for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
