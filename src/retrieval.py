"""
Retrieval Approaches for RAG Pipeline

This module implements multiple retrieval strategies:
1. Dense Retrieval - Using sentence transformer embeddings + FAISS
2. Sparse Retrieval - BM25 (lexical matching)
3. Hybrid Retrieval - Combining dense and sparse with score fusion
4. Re-ranking - Cross-encoder based re-ranking for improved precision

The hybrid approach typically provides the best results by leveraging
both semantic similarity (dense) and exact term matching (sparse).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math
from collections import Counter
import re

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from .chunking import Chunk


@dataclass
class RetrievalResult:
    """A single retrieval result with scoring details."""
    chunk: Chunk
    score: float
    rank: int
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


class DenseRetriever:
    """
    Dense retrieval using sentence transformers and FAISS.
    
    Uses semantic embeddings to find chunks that are conceptually similar
    to the query, even if they don't share exact terms.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build FAISS index from chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        
        # Generate embeddings
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity for normalized vectors
        self.index.add(self.embeddings)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call index_chunks first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        
        return results


class BM25Retriever:
    """
    BM25 sparse retrieval (lexical matching).
    
    Uses term frequency and inverse document frequency to find
    chunks with matching terms. Good for exact keyword matching.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.tokenized_docs: List[List[str]] = []
        self.idf: Dict[str, float] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split, remove punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'this', 'that', 'these', 'those',
                     'it', 'its'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self.tokenized_docs = []
        self.doc_freqs = {}
        
        # Tokenize all documents
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.tokenized_docs.append(tokens)
            
            # Count document frequencies (how many docs contain each term)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Pre-compute IDF
        N = len(chunks)
        for term, df in self.doc_freqs.items():
            # IDF with smoothing
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def _score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a single document."""
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Term frequencies in document
        tf = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue
            
            idf = self.idf[term]
            
            # BM25 formula
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator
        
        return score
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k chunks for a query."""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Score all documents
        scores = []
        for idx in range(len(self.chunks)):
            score = self._score_document(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.chunks[idx], score))
        
        return results


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    retrievers, getting the best of both worlds:
    - Dense: semantic similarity, handles paraphrasing
    - Sparse: exact term matching, handles specific keywords
    """
    
    def __init__(
        self,
        dense_model: str = "all-MiniLM-L6-v2",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ):
        self.dense_retriever = DenseRetriever(model_name=dense_model)
        self.sparse_retriever = BM25Retriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k  # RRF constant
        self.chunks: List[Chunk] = []
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build both dense and sparse indices."""
        self.chunks = chunks
        self.dense_retriever.index_chunks(chunks)
        self.sparse_retriever.index_chunks(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fusion_method: str = "rrf"
    ) -> List[RetrievalResult]:
        """
        Retrieve and fuse results from both retrievers.
        
        fusion_method: "rrf" (Reciprocal Rank Fusion) or "score" (weighted sum)
        """
        # Get results from both retrievers (fetch more for fusion)
        fetch_k = top_k * 3
        dense_results = self.dense_retriever.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=fetch_k)
        
        if fusion_method == "rrf":
            return self._rrf_fusion(dense_results, sparse_results, top_k)
        else:
            return self._score_fusion(dense_results, sparse_results, top_k)
    
    def _rrf_fusion(
        self,
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion."""
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"chunk": chunk, "rrf_score": 0, "dense_score": None, "sparse_score": None}
            chunk_scores[chunk_id]["rrf_score"] += self.dense_weight / (self.rrf_k + rank)
            chunk_scores[chunk_id]["dense_score"] = score
        
        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"chunk": chunk, "rrf_score": 0, "dense_score": None, "sparse_score": None}
            chunk_scores[chunk_id]["rrf_score"] += self.sparse_weight / (self.rrf_k + rank)
            chunk_scores[chunk_id]["sparse_score"] = score
        
        # Sort by RRF score
        sorted_results = sorted(chunk_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        # Convert to RetrievalResult objects
        results = []
        for rank, item in enumerate(sorted_results[:top_k], start=1):
            results.append(RetrievalResult(
                chunk=item["chunk"],
                score=item["rrf_score"],
                rank=rank,
                dense_score=item["dense_score"],
                sparse_score=item["sparse_score"]
            ))
        
        return results
    
    def _score_fusion(
        self,
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        top_k: int
    ) -> List[RetrievalResult]:
        """Weighted score fusion with normalization."""
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        
        # Normalize dense scores
        if dense_results:
            max_dense = max(score for _, score in dense_results)
            min_dense = min(score for _, score in dense_results)
            dense_range = max_dense - min_dense if max_dense != min_dense else 1
            
            for chunk, score in dense_results:
                norm_score = (score - min_dense) / dense_range
                chunk_id = chunk.chunk_id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {"chunk": chunk, "score": 0, "dense_score": None, "sparse_score": None}
                chunk_scores[chunk_id]["score"] += self.dense_weight * norm_score
                chunk_scores[chunk_id]["dense_score"] = score
        
        # Normalize sparse scores
        if sparse_results:
            max_sparse = max(score for _, score in sparse_results)
            min_sparse = min(score for _, score in sparse_results)
            sparse_range = max_sparse - min_sparse if max_sparse != min_sparse else 1
            
            for chunk, score in sparse_results:
                norm_score = (score - min_sparse) / sparse_range
                chunk_id = chunk.chunk_id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {"chunk": chunk, "score": 0, "dense_score": None, "sparse_score": None}
                chunk_scores[chunk_id]["score"] += self.sparse_weight * norm_score
                chunk_scores[chunk_id]["sparse_score"] = score
        
        # Sort by combined score
        sorted_results = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        
        results = []
        for rank, item in enumerate(sorted_results[:top_k], start=1):
            results.append(RetrievalResult(
                chunk=item["chunk"],
                score=item["score"],
                rank=rank,
                dense_score=item["dense_score"],
                sparse_score=item["sparse_score"]
            ))
        
        return results


class ReRanker:
    """
    Cross-encoder based re-ranker for improved precision.
    
    Takes initial retrieval results and re-scores them using a more
    powerful cross-encoder model that looks at query-document pairs together.
    
    This is slower but more accurate than bi-encoder retrieval.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Re-rank results using cross-encoder."""
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.chunk.text) for r in results]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results, start=1):
            result.rank = i
            result.score = result.rerank_score  # Use rerank score as final score
        
        if top_k:
            results = results[:top_k]
        
        return results


class RAGRetriever:
    """
    Full-featured RAG retriever with all capabilities.
    
    Combines:
    - Hybrid retrieval (dense + BM25)
    - Optional re-ranking
    - Configurable fusion methods
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranking: bool = True,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        self.hybrid_retriever = HybridRetriever(
            dense_model=embedding_model,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        self.use_reranking = use_reranking
        self.reranker = ReRanker(rerank_model) if use_reranking and rerank_model else None
        self.chunks: List[Chunk] = []
    
    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build retrieval index from chunks."""
        self.chunks = chunks
        self.hybrid_retriever.index_chunks(chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: Optional[int] = None,
        fusion_method: str = "rrf"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of final results to return
            rerank_top_k: Number of candidates to rerank (default: top_k * 3)
            fusion_method: "rrf" or "score"
        
        Returns:
            List of RetrievalResult objects
        """
        # Retrieve more candidates for reranking
        fetch_k = rerank_top_k or (top_k * 3 if self.use_reranking else top_k)
        
        results = self.hybrid_retriever.retrieve(query, top_k=fetch_k, fusion_method=fusion_method)
        
        # Apply re-ranking if enabled
        if self.use_reranking and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve with detailed scoring breakdown."""
        results = self.retrieve(query, top_k=top_k)
        
        return [
            {
                "text": r.chunk.text,
                "chunk_id": r.chunk.chunk_id,
                "page": r.chunk.source_page,
                "strategy": r.chunk.strategy,
                "rank": r.rank,
                "final_score": r.score,
                "dense_score": r.dense_score,
                "sparse_score": r.sparse_score,
                "rerank_score": r.rerank_score
            }
            for r in results
        ]


