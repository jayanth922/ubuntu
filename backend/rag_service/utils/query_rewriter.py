import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer

class HybridSearchEngine:
    """Hybrid search combining dense and sparse retrieval"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index for dense retrieval
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize BM25 for sparse retrieval
        self.bm25 = None
        self.documents = []
        self.doc_ids_map = {}  # Maps index positions to document IDs
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = text.split()
        # Filter stopwords (simplified)
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'in', 'to', 'of', 'for', 'with'}
        return [token for token in tokens if token not in stopwords]
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents for both dense and sparse retrieval"""
        if not documents:
            return
        
        self.documents = documents
        
        # Index for dense retrieval (FAISS)
        embeddings = []
        for i, doc in enumerate(documents):
            if 'content' in doc:
                embeddings.append(self.model.encode(doc['content']))
                self.doc_ids_map[i] = doc.get('id') or doc.get('chunk_id') or str(i)
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings_array)
        
        # Index for sparse retrieval (BM25)
        tokenized_corpus = [
            self._preprocess_text(doc.get('content', '')) 
            for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight between dense (1.0) and sparse (0.0) retrieval
        """
        if not self.documents:
            return []
        
        # Ensure top_k doesn't exceed document count
        top_k = min(top_k, len(self.documents))
        
        # Get dense retrieval results
        query_embedding = self.model.encode(query).reshape(1, -1).astype('float32')
        distances, dense_indices = self.index.search(query_embedding, top_k * 2)  # Get more to merge
        
        # Normalize distances to scores (lower distance -> higher score)
        max_dist = np.max(distances) if distances.size > 0 else 1.0
        min_dist = np.min(distances) if distances.size > 0 else 0.0
        dense_scores = 1.0 - (distances - min_dist) / (max_dist - min_dist + 1e-6)
        
        # Get sparse retrieval results
        tokenized_query = self._preprocess_text(query)
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_score = np.max(sparse_scores) if sparse_scores.size > 0 else 1.0
        min_score = np.min(sparse_scores) if sparse_scores.size > 0 else 0.0
        sparse_scores = (sparse_scores - min_score) / (max_score - min_score + 1e-6)
        
        # Combine results with weighted score
        combined_scores = {}
        
        # Process dense results
        for i, idx in enumerate(dense_indices[0]):
            if idx < len(self.documents):
                doc_id = self.doc_ids_map.get(idx, str(idx))
                score = float(dense_scores[0][i])
                combined_scores[doc_id] = {'score': alpha * score, 'doc_idx': idx}
        
        # Process sparse results
        for idx, score in enumerate(sparse_scores):
            if idx < len(self.documents):
                doc_id = self.doc_ids_map.get(idx, str(idx))
                if doc_id in combined_scores:
                    combined_scores[doc_id]['score'] += (1 - alpha) * score
                else:
                    combined_scores[doc_id] = {'score': (1 - alpha) * score, 'doc_idx': idx}
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:top_k]
        
        # Return documents with scores
        results = []
        for doc_id, info in sorted_results:
            doc_idx = info['doc_idx']
            if 0 <= doc_idx < len(self.documents):
                doc = self.documents[doc_idx].copy()
                doc['similarity_score'] = float(info['score'])
                results.append(doc)
        
        return results