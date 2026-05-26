from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# What are Embeddings?
# Embeddings are mathematical representations (vectors or arrays of numbers) of text.
# They capture the semantic meaning of the text. Words or sentences with similar meanings
# will have similar embeddings, and they will be "closer" to each other in a mathematical space.

# Why are Embeddings used in RAG?
# In RAG (Retrieval-Augmented Generation), we need a way to quickly find the most relevant
# pieces of information (chunks) from our uploaded documents that answer the user's question.
# By converting both the chunks and the user's question into embeddings, we can measure the
# "distance" between them (e.g., using Cosine Similarity). The chunks with embeddings closest 
# to the question's embedding are the most relevant and are retrieved as context for the LLM.

def get_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Initializes and returns the SentenceTransformer model.
    'all-MiniLM-L6-v2' is a lightweight and fast model good for semantic search.
    """
    return SentenceTransformer(model_name)

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """
    Generates embeddings for a list of strings.
    
    Args:
        texts: A list of text chunks.
        model: The initialized SentenceTransformer model.
        
    Returns:
        A list of embedding vectors (lists of floats).
    """
    if not texts:
        return []
    
    # model.encode returns a numpy array, we convert it to a list of floats for ChromaDB
    embeddings = model.encode(texts)
    return embeddings.tolist()
