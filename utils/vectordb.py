import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

# We've replaced ChromaDB with a lightweight Numpy-based Vector Database.
# This prevents compatibility issues (like OpenTelemetry/protobuf) on Python 3.14
# while perfectly maintaining the "Similarity Search" functionality needed for our RAG.

DB_FILE = "./simple_vectordb.json"

class SimpleVectorDB:
    def __init__(self, filepath=DB_FILE):
        self.filepath = filepath
        self.data = self._load()
        
    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}
        
    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f)
            
    def add(self, ids, embeddings, documents, metadatas):
        self.data["ids"].extend(ids)
        self.data["embeddings"].extend(embeddings)
        self.data["texts"].extend(documents)
        self.data["metadatas"].extend(metadatas)
        self._save()
        
    def query(self, query_embeddings, n_results, include=None, where=None):
        if not self.data["embeddings"]:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        q_emb = np.array(query_embeddings[0])
        db_embs = np.array(self.data["embeddings"])
        
        # Calculate Cosine Similarity
        q_norm = np.linalg.norm(q_emb)
        db_norms = np.linalg.norm(db_embs, axis=1)
        
        db_norms[db_norms == 0] = 1e-10
        if q_norm == 0:
            q_norm = 1e-10
            
        similarities = np.dot(db_embs, q_emb) / (db_norms * q_norm)
        
        # Distance = 1 - similarity (so lower distance means more similar, matching ChromaDB format)
        distances = 1.0 - similarities
        
        # Metadata Filtering
        valid_indices = []
        for i, meta in enumerate(self.data["metadatas"]):
            if where:
                match = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if match:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)
                
        if not valid_indices:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        filtered_distances = [(distances[i], i) for i in valid_indices]
        filtered_distances.sort(key=lambda x: x[0])
        
        top_k = filtered_distances[:n_results]
        
        res_docs = []
        res_metas = []
        res_dists = []
        
        for dist, idx in top_k:
            res_docs.append(self.data["texts"][idx])
            res_metas.append(self.data["metadatas"][idx])
            res_dists.append(float(dist))
            
        return {
            "documents": [res_docs],
            "metadatas": [res_metas],
            "distances": [res_dists]
        }

    def delete_collection(self):
        self.data = {"ids": [], "texts": [], "metadatas": [], "embeddings": []}
        self._save()


# --- Maintain original function signatures so the rest of the app doesn't break ---

def get_chroma_client(path: str = "./chroma_db"):
    return None  # We don't need a persistent client object anymore

def get_or_create_collection(client, collection_name: str = "rag_documents"):
    return SimpleVectorDB()

def store_chunks_in_db(collection, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    if not chunks:
        return
    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

def similarity_search(collection, query_embedding: List[float], n_results: int = 3, metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    if metadata_filter:
        search_params["where"] = metadata_filter
    return collection.query(**search_params)

def clear_database(client, collection_name: str = "rag_documents"):
    db = SimpleVectorDB()
    db.delete_collection()
