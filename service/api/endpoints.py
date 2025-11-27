from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os
import logging
import csv
from datetime import datetime
from service.core.config import settings
from service.rag.retriever import Retriever
from service.rag.generator import Generator
from ingest.parser import PDFParser
from ingest.cleaner import TextCleaner
from ingest.splitter import TextSplitter
from ingest.embedder import EmbeddingGenerator
from ingest.indexer import ChromaIndexer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize components (Lazy loading or global init could be better for prod, but this is fine for now)
retriever = Retriever(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, collection_name=settings.CHROMA_COLLECTION)
generator = Generator(model_name=settings.LLM_MODEL)

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class FeedbackRequest(BaseModel):
    query_id: str
    feedback: str
    rating: int

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # 1. Retrieve
        documents = retriever.retrieve(request.query, k=request.k)
        
        # 2. Generate
        answer = generator.generate(request.query, documents)
        
        return QueryResponse(answer=answer, sources=documents)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve")
async def retrieve(request: QueryRequest):
    try:
        documents = retriever.retrieve(request.query, k=request.k)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_ingestion(file_path: str):
    try:
        logger.info(f"Processing {file_path}...")
        parser = PDFParser()
        documents = parser.parse(file_path)
        
        cleaner = TextCleaner()
        for doc in documents:
            doc["text"] = cleaner.clean(doc["text"])
            
        splitter = TextSplitter()
        chunked_docs = splitter.split(documents)
        
        embedder = EmbeddingGenerator(model_name=settings.EMBEDDING_MODEL)
        texts = [doc["text"] for doc in chunked_docs]
        embeddings = embedder.embed_documents(texts)
        
        indexer = ChromaIndexer(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, collection_name=settings.CHROMA_COLLECTION)
        indexer.index_documents(chunked_docs, embeddings)
        logger.info(f"Finished processing {file_path}")
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")

@router.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_location = f"ingest/temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    background_tasks.add_task(process_ingestion, file_location)
    
    return {"message": "Ingestion started in background", "filename": file.filename}

@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    # Save to CSV
    feedback_file = "feedback.csv"
    file_exists = os.path.isfile(feedback_file)
    
    try:
        with open(feedback_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "query_id", "rating", "feedback"])
            
            writer.writerow([datetime.utcnow().isoformat(), request.query_id, request.rating, request.feedback])
            
        logger.info(f"Saved feedback for {request.query_id}")
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@router.get("/health")
async def health():
    return {"status": "ok"}
