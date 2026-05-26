import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any, Optional

# A Vector Database is a specialized database designed to store, manage, and search embeddings (vectors).
# ChromaDB is used here because it is open-source, runs locally, and is easy to integrate.
# It allows us to perform "Similarity Search", where we find vectors in the database that are
# closest to our query vector, typically using a metric like "Cosine Similarity".

def get_chroma_client(path: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    Initializes and returns a ChromaDB PersistentClient.
    Persistent means data is saved to disk so it survives app restarts.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return chromadb.PersistentClient(path=path)

def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = "rag_documents"):
    """
    Gets an existing collection or creates a new one.
    A collection in ChromaDB is similar to a table in a relational database.
    """
    # We use cosine similarity to measure how close vectors are to each other.
    # A higher cosine similarity means the texts are semantically more similar.
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def store_chunks_in_db(
    collection,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
):
    """
    Stores chunks, their embeddings, metadata, and unique IDs in the vector database.
    """
    if not chunks:
        return

    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Add data to the ChromaDB collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

def similarity_search(
    collection, 
    query_embedding: List[float], 
    n_results: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs a Similarity Search to find the top `n_results` chunks most similar to the query.
    
    Similarity Search works by calculating the mathematical distance (cosine similarity) 
    between the query embedding and all stored chunk embeddings.
    
    Metadata Filtering (Key-Value Search) allows us to restrict the search to chunks
    that match specific criteria (e.g., only chunks from a specific file).
    """
    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }
    
    # If a metadata filter is provided (e.g., {"filename": "policy.pdf"})
    if metadata_filter:
        search_params["where"] = metadata_filter

    results = collection.query(**search_params)
    return results

def clear_database(client: chromadb.PersistentClient, collection_name: str = "rag_documents"):
    """
    Deletes the collection to clear all stored data.
    """
    try:
        client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Error clearing collection: {e}")
