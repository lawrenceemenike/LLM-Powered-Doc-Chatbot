import argparse
import os
import logging
from ingest.parser import PDFParser
from ingest.cleaner import TextCleaner
from ingest.splitter import TextSplitter
from ingest.embedder import EmbeddingGenerator
from ingest.indexer import ChromaIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file to ingest.")
    parser.add_argument("--chroma_host", type=str, default="localhost", help="ChromaDB host.")
    parser.add_argument("--chroma_port", type=int, default=8000, help="ChromaDB port.")
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    logger.info(f"Parsing {file_path}...")
    pdf_parser = PDFParser()
    documents = pdf_parser.parse(file_path)
    
    logger.info("Cleaning text...")
    cleaner = TextCleaner()
    for doc in documents:
        doc["text"] = cleaner.clean(doc["text"])

    logger.info("Splitting text...")
    splitter = TextSplitter()
    chunked_docs = splitter.split(documents)
    
    logger.info("Generating embeddings...")
    embedder = EmbeddingGenerator()
    texts = [doc["text"] for doc in chunked_docs]
    embeddings = embedder.embed_documents(texts)
    
    logger.info("Indexing into ChromaDB...")
    indexer = ChromaIndexer(host=args.chroma_host, port=args.chroma_port)
    indexer.index_documents(chunked_docs, embeddings)
    
    logger.info("Ingestion complete.")

if __name__ == "__main__":
    main()
