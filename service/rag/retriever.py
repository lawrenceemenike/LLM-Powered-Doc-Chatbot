import chromadb
from typing import List, Dict
from ingest.embedder import EmbeddingGenerator

class Retriever:
    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "documents"):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedder = EmbeddingGenerator()

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieves top-k documents for a query.
        
        Args:
            query: The query string.
            k: Number of documents to retrieve.
            
        Returns:
            List of retrieved documents with metadata and distance.
        """
        query_embedding = self.embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                doc = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None
                }
                documents.append(doc)
                
        return documents
