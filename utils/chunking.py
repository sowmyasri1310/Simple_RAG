from typing import List, Dict, Any
import uuid
import time

def split_into_chunks(extracted_data: List[Dict[str, Any]], chunk_size_words: int = 500, overlap_words: int = 50) -> List[Dict[str, Any]]:
    """
    Splits extracted text into chunks of approximately `chunk_size_words` with `overlap_words`.
    Preserves metadata such as filename, page number, and adds chunk ID and upload timestamp.
    
    Args:
        extracted_data: List of dicts containing text and metadata from pdf_loader.
        chunk_size_words: Target number of words per chunk.
        overlap_words: Number of overlapping words between consecutive chunks.
        
    Returns:
        A list of dictionaries representing the chunks and their metadata.
    """
    chunks = []
    upload_timestamp = int(time.time())
    
    for doc in extracted_data:
        text = doc["text"]
        filename = doc["filename"]
        page_number = doc["page_number"]
        
        words = text.split()
        
        if not words:
            continue
            
        i = 0
        while i < len(words):
            # Take a slice of words up to chunk_size_words
            chunk_words = words[i:i + chunk_size_words]
            chunk_text = " ".join(chunk_words)
            
            # Create a unique ID for each chunk
            chunk_id = str(uuid.uuid4())
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "upload_timestamp": upload_timestamp
                }
            })
            
            # Move the index forward, accounting for the overlap
            i += (chunk_size_words - overlap_words)
            
            # Safety check to avoid infinite loop if chunk_size <= overlap
            if chunk_size_words <= overlap_words:
                break

    return chunks
