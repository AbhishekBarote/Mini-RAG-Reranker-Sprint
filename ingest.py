import sqlite3
import json
import os
import logging
import re
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def sanitize_filename(title):
    """
    Sanitize the title to create a valid filename by removing or replacing invalid characters.
    """
    # Replace characters that are problematic in filenames
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = sanitized.replace(' ', '_').replace('/', '_').replace('\\', '_')
    # Limit length to avoid OS path length issues
    return sanitized[:100] + '.pdf'

def find_pdf_file(documents_dir, title):
    """
    Attempt to find a PDF file that matches the title from sources.json.
    Checks for exact matches, sanitized matches, and partial matches.
    """
    # First try: exact match
    exact_path = os.path.join(documents_dir, f"{title}.pdf")
    if os.path.exists(exact_path):
        return exact_path
    
    # Second try: sanitized filename
    sanitized_title = sanitize_filename(title)
    sanitized_path = os.path.join(documents_dir, sanitized_title)
    if os.path.exists(sanitized_path):
        return sanitized_path
    
    # Third try: look for any PDF that contains the title (or part of it)
    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        # Check if title is contained in the filename (case-insensitive)
        if title.lower() in pdf_file.lower():
            return os.path.join(documents_dir, pdf_file)
    
    # Final try: check without the file extension in the title
    if title.endswith('.pdf'):
        title_without_ext = title[:-4]
        return find_pdf_file(documents_dir, title_without_ext)
    
    return None

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start position, considering overlap
        start += chunk_size - chunk_overlap
    
    return chunks

def create_database():
    """Create SQLite database with chunks table if it doesn't exist."""
    conn = sqlite3.connect('chunks.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source_title TEXT,
            source_url TEXT,
            text TEXT,
            embedding BLOB,
            chunk_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_source ON chunks (source_title);
    ''')
    conn.commit()
    conn.close()

def process_documents():
    """Process all PDF documents and store chunks in SQLite database."""
    # Create database
    create_database()
    
    # Create database connection
    conn = sqlite3.connect('chunks.db')
    cursor = conn.cursor()
    
    # Load sources
    try:
        with open('sources.json', 'r', encoding='utf-8') as f:
            sources = json.load(f)
        logger.info(f"Loaded {len(sources)} sources from sources.json")
    except FileNotFoundError:
        logger.error("sources.json not found in the current directory.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing sources.json: {e}")
        return
    
    # Create documents directory if it doesn't exist
    documents_dir = 'documents'
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        logger.warning(f"Created {documents_dir} directory. Please add PDF files to it.")
        return
    
    # Verify we have some PDFs to process
    pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {documents_dir} directory.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files in {documents_dir} directory.")
    
    # Process each document
    total_chunks = 0
    processed_count = 0
    
    for source in sources:
        title = source.get('title', '')
        url = source.get('url', '')
        
        if not title:
            logger.warning("Skipping source with missing title.")
            continue
        
        pdf_path = find_pdf_file(documents_dir, title)
        
        if not pdf_path:
            logger.warning(f"Could not find PDF for: {title}")
            continue
        
        try:
            logger.info(f"Processing: {title}")
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logger.debug(f"Page {page_num + 1} in {title} contained no text.")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} in {title}: {e}")
            
            if not text.strip():
                logger.warning(f"No text extracted from {title}. The PDF might be scanned or encrypted.")
                continue
            
            # Split into chunks
            chunks = chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from {title}")
            
            # Store each chunk
            for i, chunk in enumerate(chunks):
                try:
                    embedding = model.encode(chunk)
                    embedding_bytes = embedding.tobytes()
                    
                    cursor.execute(
                        "INSERT INTO chunks (source_title, source_url, text, embedding, chunk_index) VALUES (?, ?, ?, ?, ?)",
                        (title, url, chunk, embedding_bytes, i)
                    )
                    total_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i} from {title}: {e}")
            
            processed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {title}: {str(e)}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Processing complete. Successfully processed {processed_count} out of {len(sources)} documents.")
    logger.info(f"Stored {total_chunks} chunks in database.")

def verify_documents():
    """Verify that documents in sources.json exist and can be processed."""
    try:
        with open('sources.json', 'r', encoding='utf-8') as f:
            sources = json.load(f)
    except FileNotFoundError:
        logger.error("sources.json not found")
        return
    
    documents_dir = 'documents'
    if not os.path.exists(documents_dir):
        logger.warning(f"{documents_dir} directory does not exist.")
        return
    
    missing_files = []
    problematic_files = []
    
    for source in sources:
        title = source.get('title', '')
        if not title:
            continue
            
        pdf_path = find_pdf_file(documents_dir, title)
        
        if not pdf_path:
            missing_files.append(title)
        else:
            # Try to open the PDF to check if it's valid
            try:
                reader = PdfReader(pdf_path)
                if len(reader.pages) == 0:
                    problematic_files.append((title, "PDF has no pages"))
            except Exception as e:
                problematic_files.append((title, f"Error reading PDF: {str(e)}"))
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} PDF files:")
        for title in missing_files:
            logger.warning(f"  - {title}")
    
    if problematic_files:
        logger.warning(f"Found {len(problematic_files)} potentially problematic files:")
        for title, issue in problematic_files:
            logger.warning(f"  - {title}: {issue}")
    
    if not missing_files and not problematic_files:
        logger.info("All documents verified successfully!")
    
    return missing_files, problematic_files

if __name__ == "__main__":
    logger.info("Starting document ingestion...")
    
    # First verify documents
    logger.info("Verifying documents...")
    missing, problematic = verify_documents()
    
    if missing:
        logger.warning("Some documents are missing. Processing will continue with available documents.")
    
    # Process available documents
    process_documents()
    logger.info("Ingestion completed!")