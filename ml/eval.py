import argparse
from typing import List, Dict
from ingest.embedder import EmbeddingGenerator
from ingest.indexer import ChromaIndexer
import chromadb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    intersection = retrieved_set.intersection(relevant_set)
    return len(intersection) / len(relevant_set) if relevant_set else 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG performance.")
    parser.add_argument("--chroma_host", type=str, default="localhost", help="ChromaDB host.")
    parser.add_argument("--chroma_port", type=int, default=8000, help="ChromaDB port.")
    args = parser.parse_args()

    # Dummy test set for demonstration
    test_set = [
        {
            "query": "What is the vendor NDA?",
            "relevant_docs": ["contracts-2025-01-23-v1_p12_c3"] # Example chunk ID
        }
    ]

    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    collection = client.get_collection(name="documents")
    embedder = EmbeddingGenerator()

    total_recall = 0.0
    for item in test_set:
        query = item["query"]
        relevant_ids = item["relevant_docs"]
        
        query_embedding = embedder.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        
        retrieved_ids = results["ids"][0] if results["ids"] else []
        recall = calculate_recall_at_k(retrieved_ids, relevant_ids, k=5)
        total_recall += recall
        
        logger.info(f"Query: {query}, Recall@5: {recall}")

    avg_recall = total_recall / len(test_set) if test_set else 0.0
    logger.info(f"Average Recall@5: {avg_recall}")

if __name__ == "__main__":
    main()
