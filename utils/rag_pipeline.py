from typing import List, Any, Dict, Tuple, Optional
from utils.pdf_loader import extract_text_from_pdfs
from utils.chunking import split_into_chunks
from utils.embeddings import get_embedding_model, generate_embeddings
from utils.vectordb import get_chroma_client, get_or_create_collection, store_chunks_in_db, similarity_search
from utils.groq_helper import get_groq_client, generate_rag_answer

# Initialize global singletons to avoid reloading models/clients
_embedding_model = get_embedding_model()
_chroma_client = get_chroma_client()
_collection = get_or_create_collection(_chroma_client)

def ingest_documents(uploaded_files: List[Any]) -> Tuple[bool, str, int]:
    """
    Orchestrates the ingestion pipeline:
    1. Extract text from PDFs
    2. Split text into chunks
    3. Generate embeddings for chunks
    4. Store in ChromaDB
    
    Returns:
        (success_boolean, message_string, number_of_chunks_processed)
    """
    if not uploaded_files:
        return False, "No files provided.", 0

    try:
        # Step 1: Extract
        extracted_data = extract_text_from_pdfs(uploaded_files)
        if not extracted_data:
            return False, "Could not extract text from the provided PDFs. They might be empty or corrupted.", 0
            
        # Step 2: Chunking
        chunks = split_into_chunks(extracted_data)
        if not chunks:
            return False, "Failed to create chunks from the extracted text.", 0
            
        # Step 3: Embedding
        texts_to_embed = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts_to_embed, _embedding_model)
        
        # Step 4: Storage
        store_chunks_in_db(_collection, chunks, embeddings)
        
        return True, f"Successfully ingested {len(uploaded_files)} file(s).", len(chunks)
        
    except Exception as e:
        return False, f"An error occurred during ingestion: {e}", 0

def query_rag_pipeline(
    user_query: str, 
    model_name: str, 
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[Dict[str, Any]], List[float]]:
    """
    Orchestrates the retrieval and generation pipeline:
    1. Embed user query
    2. Perform similarity search in ChromaDB
    3. Generate answer using Groq LLM
    
    Returns:
        (generated_answer, retrieved_chunks_metadata, similarity_scores)
    """
    try:
        # Step 1: Embed user query
        query_embedding = generate_embeddings([user_query], _embedding_model)[0]
        
        # Step 2: Similarity search
        search_results = similarity_search(
            _collection, 
            query_embedding, 
            n_results=3, 
            metadata_filter=metadata_filter
        )
        
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]
        
        if not documents:
            return "No relevant information found in the database. Please upload some documents.", [], []
            
        # Compile retrieved chunks for the UI explainability feature
        retrieved_info = []
        for doc, meta in zip(documents, metadatas):
            retrieved_info.append({
                "text": doc,
                "metadata": meta
            })
            
        # Step 3: Generate answer using Groq
        try:
            groq_client = get_groq_client()
        except ValueError as e:
            return str(e), retrieved_info, distances

        answer = generate_rag_answer(
            client=groq_client,
            user_question=user_query,
            retrieved_chunks=documents,
            model=model_name
        )
        
        return answer, retrieved_info, distances

    except Exception as e:
        return f"An error occurred during retrieval/generation: {e}", [], []

def clear_system_database() -> bool:
    """
    Clears the ChromaDB collection and recreates it.
    """
    from utils.vectordb import clear_database
    try:
        clear_database(_chroma_client)
        global _collection
        _collection = get_or_create_collection(_chroma_client)
        return True
    except Exception as e:
        print(f"Error recreating collection: {e}")
        return False
