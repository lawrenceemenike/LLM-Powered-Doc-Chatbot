# LLM-Powered Doc Chatbot (RAG + MLOps)

A production-ready, open-source retrieval-augmented generation (RAG) document chatbot.

## Features
- **Ingestion**: PDF parsing, text cleaning, chunking, and embedding.
- **RAG**: Semantic search with ChromaDB and LLM generation.
- **API**: FastAPI service for querying and ingestion.
- **MLOps**: CI/CD pipelines, evaluation scripts, and monitoring with Prometheus/Grafana.

## Quickstart

### Prerequisites
- Docker & Docker Compose
- Python 3.9+

### Run Locally
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "LLM-Powered Doc Chatbot (MLOps)"
   ```

2. **Start the stack:**
   ```bash
   docker-compose up --build
   ```

3. **Ingest a document:**
   ```bash
   # In a new terminal
   docker-compose exec app python ingest/cli.py --file /app/ingest/sample.pdf
   # Note: Place a sample.pdf in the ingest/ directory first.
   ```

4. **Query the chatbot:**
   Open your browser to `http://localhost:8000/docs` and try the `/query` endpoint.
   ```json
   {
     "query": "What is the summary of the document?",
     "k": 5
   }
   ```

## Architecture
See [docs/architecture.md](docs/architecture.md) for details.

## Development
- **Run Unit Tests:** `pytest tests/unit`
- **Run Integration Tests:** `pytest tests/integration`
