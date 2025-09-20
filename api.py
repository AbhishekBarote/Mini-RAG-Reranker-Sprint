from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
import json
from typing import List, Optional, Dict, Any
import re
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mini RAG API",
    description="A lightweight Retrieval-Augmented Generation API for industrial safety documents",
    version="1.0.0"
)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Request models
class Query(BaseModel):
    q: str
    k: int = 5
    mode: str = "hybrid"  # "baseline" or "hybrid"
    alpha: float = 0.6    # Weight for vector score in hybrid mode

# Response models
class Context(BaseModel):
    score: float
    text: str
    source_title: str
    source_url: str
    chunk_index: int

class Answer(BaseModel):
    answer: str
    sources: List[str]

class Response(BaseModel):
    answer: Optional[Answer] = None
    contexts: List[Context]
    reranker_used: bool
    abstention_reason: Optional[str] = None
    processing_time: float

class Document(BaseModel):
    title: str
    url: str
    text_content: str

def get_db_connection():
    conn = sqlite3.connect('chunks.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_all_chunks():
    """Retrieve all chunks from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        chunks = cursor.execute("SELECT id, source_title, source_url, text, embedding, chunk_index FROM chunks").fetchall()
        conn.close()
        logger.info(f"Retrieved {len(chunks)} chunks from database")
        return chunks
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return None

def normalize_scores(scores):
    """Normalize scores to 0-1 range"""
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)  # All same, return middle value
    
    return [(score - min_score) / (max_score - min_score) for score in scores]

def clean_text(text):
    """Clean text by removing excessive whitespace and special characters"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_answer(text, max_length=500):
    """Format the answer text with proper truncation"""
    text = clean_text(text)
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a sentence boundary
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    for sentence in sentences:
        if len(result) + len(sentence) + 1 <= max_length:
            result += sentence + " "
        else:
            break
    
    result = result.strip()
    if not result:  # If no sentences fit, just truncate
        result = text[:max_length].rsplit(' ', 1)[0] + "..."
    elif len(result) < len(text):
        result += "..."
    
    return result

def fix_embedding_dimension(embedding_bytes, target_dimension=384):
    """Fix embedding dimension if it doesn't match the expected dimension"""
    try:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        if len(embedding) == target_dimension:
            return embedding
        else:
            # Handle dimension mismatch by padding or truncating
            logger.warning(f"Embedding dimension mismatch: expected {target_dimension}, got {len(embedding)}")
            if len(embedding) > target_dimension:
                return embedding[:target_dimension]
            else:
                # Pad with zeros
                padded = np.zeros(target_dimension)
                padded[:len(embedding)] = embedding
                return padded
    except Exception as e:
        logger.error(f"Error processing embedding: {e}")
        return np.zeros(target_dimension)

@app.post("/ingest", status_code=201)
async def ingest_documents(documents: List[Document]):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, source_title TEXT, source_url TEXT, text TEXT, embedding BLOB, chunk_index INTEGER)")
        conn.commit()

        for doc in documents:
            # Simple chunking for demonstration (can be replaced with more advanced methods)
            chunks_content = [doc.text_content[i:i + 500] for i in range(0, len(doc.text_content), 500)]
            for i, chunk_text in enumerate(chunks_content):
                embedding = model.encode(chunk_text).tobytes()
                cursor.execute("INSERT INTO chunks (source_title, source_url, text, embedding, chunk_index) VALUES (?, ?, ?, ?, ?)",
                               (doc.title, doc.url, chunk_text, embedding, i))
        conn.commit()
        conn.close()
        logger.info(f"Successfully ingested {len(documents)} documents with {len(chunks_content)} chunks each.")
        return {"message": f"Successfully ingested {len(documents)} documents."}
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        conn.close()
        if count > 0:
            return {"status": "ok", "message": f"Database contains {count} chunks.", "detail": "OK"}
        else:
            return {"status": "warning", "message": "Database is empty. Run ingestion first.", "detail": "Not Found"}
    except sqlite3.Error as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection error")


@app.post("/ask", response_model=Response)
async def ask_query(query: Query):
    start_time = datetime.now()
    
    try:
        # Get all chunks from database
        chunks = get_all_chunks()
        if not chunks:
            raise HTTPException(status_code=503, detail="No chunks found in database. Please run ingestion first.")
        
        logger.info(f"Processing query: '{query.q}' with mode: {query.mode}")
        
        # Extract text, sources, and embeddings
        texts = [chunk['text'] for chunk in chunks]
        source_titles = [chunk['source_title'] for chunk in chunks]
        source_urls = [chunk['source_url'] for chunk in chunks]
        chunk_indices = [chunk['chunk_index'] for chunk in chunks]
        
        # Convert embeddings from bytes to numpy arrays with dimension handling
        embeddings = []
        for chunk in chunks:
            embedding_bytes = chunk['embedding']
            if embedding_bytes:
                embedding = fix_embedding_dimension(embedding_bytes, EMBEDDING_DIMENSION)
                embeddings.append(embedding)
            else:
                # Handle case where embedding is None or empty
                embeddings.append(np.zeros(EMBEDDING_DIMENSION))
        logger.info(f"Loaded embeddings. Example shape: {embeddings[0].shape}, dtype: {embeddings[0].dtype}")
        
        # Calculate vector similarities
        query_embedding = model.encode(query.q)
        logger.info(f"Query embedding. Shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        similarities = []
        for vec in embeddings:
            # Handle zero vectors to avoid division by zero
            vec_norm = np.linalg.norm(vec)
            query_norm = np.linalg.norm(query_embedding)
            
            logger.info(f"vec_norm: {vec_norm}, query_norm: {query_norm}")
            if float(vec_norm) == 0 or float(query_norm) == 0:
                similarity = 0
            else:
                similarity = np.dot(query_embedding, vec) / (query_norm * vec_norm)
            similarities.append(similarity)
        
        # Calculate BM25 scores for keyword matching
        # try:
        #     tokenized_corpus = [text.split() for text in texts]
        #     bm25 = BM25Okapi(tokenized_corpus)
        #     bm25_scores = bm25.get_scores(query.q.split())
        # except Exception as e:
        #     logger.warning(f"BM25 calculation failed: {e}")
        #     # If BM25 fails, use uniform scores
        #     bm25_scores = [0.5] * len(texts)
        bm25_scores = [0.5] * len(texts) # Temporarily set uniform scores
        
        # Normalize scores
        norm_similarities = normalize_scores(similarities)
        norm_bm25_scores = normalize_scores(bm25_scores)
        
        # Combine scores based on mode
        combined_scores = []
        for i, (sim_score, bm25_score) in enumerate(zip(norm_similarities, norm_bm25_scores)):
            if query.mode == "hybrid":
                combined = query.alpha * sim_score + (1 - query.alpha) * bm25_score
            else:  # baseline mode
                combined = sim_score
                
            combined_scores.append((combined, i))
        
        # Sort by combined score
        combined_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Prepare results
        results = []
        for score, idx in combined_scores[:query.k]:
            results.append(Context(
                score=float(score),
                text=texts[idx],
                source_title=source_titles[idx],
                source_url=source_urls[idx],
                chunk_index=chunk_indices[idx]
            ))
        
        # Debug: print top scores
        logger.info(f"Top {min(3, len(results))} scores: {[r.score for r in results[:3]]}")
        
        # Formulate answer if we have good results
        answer = None
        abstention_reason = None
        
        if results and results[0].score >= 0.3:  # Lowered threshold for testing
            best_match = results[0]
            answer_text = format_answer(best_match.text)
            
            answer = Answer(
                answer=answer_text,
                sources=[best_match.source_url]
            )
        else:
            abstention_reason = f"No results met the confidence threshold (0.3). Top score: {results[0].score if results else 'N/A'}"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return Response(
            answer=answer,
            contexts=results,
            reranker_used=(query.mode == "hybrid"),
            abstention_reason=abstention_reason,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ... (keep the rest of the endpoints the same as before)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")