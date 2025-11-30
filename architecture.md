# Architecture

## Overview
The system consists of the following components:

1.  **Ingestion Pipeline**:
    -   **Parser**: Extracts text from PDFs.
    -   **Cleaner**: Normalizes text.
    -   **Splitter**: Chunks text into manageable pieces.
    -   **Embedder**: Generates vector embeddings using HuggingFace models.
    -   **Indexer**: Stores embeddings and metadata in ChromaDB.

2.  **RAG Service**:
    -   **Retriever**: Queries ChromaDB for relevant context.
    -   **Generator**: Uses an LLM to generate answers based on context.
    -   **API**: FastAPI application exposing endpoints.

3.  **Infrastructure**:
    -   **Docker Compose**: Orchestrates services locally.
    -   **Prometheus/Grafana**: Monitoring and observability.

## Diagram
```mermaid
graph LR
    A[User] --> B[FastAPI Service]
    B --> C[Retriever]
    B --> D[Generator]
    C --> E[ChromaDB]
    D --> F[LLM (HF)]
    G[Ingestion CLI] --> H[PDF Parser]
    H --> I[Embedder]
    I --> E
```
