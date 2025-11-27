from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generates embedding for a single query text.
        
        Args:
            text: The query string.
            
        Returns:
            The embedding vector.
        """
        return self.embeddings.embed_query(text)
