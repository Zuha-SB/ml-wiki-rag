"""
Retrieval Evaluation Framework

This module provides comprehensive evaluation metrics for RAG retrieval:
1. Mean Reciprocal Rank (MRR) - Position of first relevant result
2. Precision@K - Proportion of relevant docs in top-K
3. Recall@K - Proportion of relevant docs found in top-K
4. NDCG@K - Normalized Discounted Cumulative Gain
5. Hit Rate@K - Whether any relevant doc is in top-K

These metrics help compare different chunking and retrieval strategies
to find the optimal configuration for a given dataset.
"""

import json
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import time

from chunking import Chunk, BaseChunker, get_chunker
from retrieval import RAGRetriever, RetrievalResult


@dataclass
class EvaluationQuery:
    """A single evaluation query with ground truth."""
    query_id: str
    query: str
    relevant_keywords: List[str]  # Keywords that should appear in relevant chunks
    expected_topics: List[str]  # High-level topics the answer should cover
    difficulty: str = "medium"  # easy, medium, hard
    
    def is_chunk_relevant(self, chunk_text: str) -> bool:
        """Check if a chunk is relevant based on keyword presence."""
        chunk_lower = chunk_text.lower()
        # A chunk is relevant if it contains at least half the keywords
        matches = sum(1 for kw in self.relevant_keywords if kw.lower() in chunk_lower)
        return matches >= max(1, len(self.relevant_keywords) // 2)


@dataclass
class QueryResult:
    """Results for a single query evaluation."""
    query_id: str
    query: str
    retrieved_chunks: List[RetrievalResult]
    relevant_indices: List[int]  # Indices of relevant chunks in retrieved results
    mrr: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    hit_at_k: Dict[int, bool]
    latency_ms: float


@dataclass
class EvaluationReport:
    """Complete evaluation report for a retrieval configuration."""
    config_name: str
    chunking_strategy: str
    retrieval_params: Dict[str, Any]
    query_results: List[QueryResult]
    
    # Aggregated metrics
    mean_mrr: float = 0.0
    mean_precision_at_k: Dict[int, float] = field(default_factory=dict)
    mean_recall_at_k: Dict[int, float] = field(default_factory=dict)
    mean_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    
    num_chunks: int = 0
    avg_chunk_size: float = 0.0
    
    def compute_aggregates(self, k_values: List[int] = [1, 3, 5, 10]):
        """Compute aggregate metrics from query results."""
        if not self.query_results:
            return
        
        n = len(self.query_results)
        
        # MRR
        self.mean_mrr = sum(qr.mrr for qr in self.query_results) / n
        
        # Precision, Recall, NDCG, Hit Rate at K
        for k in k_values:
            self.mean_precision_at_k[k] = sum(qr.precision_at_k.get(k, 0) for qr in self.query_results) / n
            self.mean_recall_at_k[k] = sum(qr.recall_at_k.get(k, 0) for qr in self.query_results) / n
            self.mean_ndcg_at_k[k] = sum(qr.ndcg_at_k.get(k, 0) for qr in self.query_results) / n
            self.hit_rate_at_k[k] = sum(1 for qr in self.query_results if qr.hit_at_k.get(k, False)) / n
        
        # Latency
        self.mean_latency_ms = sum(qr.latency_ms for qr in self.query_results) / n
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_name": self.config_name,
            "chunking_strategy": self.chunking_strategy,
            "retrieval_params": self.retrieval_params,
            "num_chunks": self.num_chunks,
            "avg_chunk_size": round(self.avg_chunk_size, 1),
            "metrics": {
                "mean_mrr": round(self.mean_mrr, 4),
                "precision@k": {k: round(v, 4) for k, v in self.mean_precision_at_k.items()},
                "recall@k": {k: round(v, 4) for k, v in self.mean_recall_at_k.items()},
                "ndcg@k": {k: round(v, 4) for k, v in self.mean_ndcg_at_k.items()},
                "hit_rate@k": {k: round(v, 4) for k, v in self.hit_rate_at_k.items()},
                "mean_latency_ms": round(self.mean_latency_ms, 2)
            },
            "num_queries_evaluated": len(self.query_results)
        }


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.
    
    Supports:
    - Multiple K values for ranking metrics
    - Relevance judgments based on keyword matching
    - Latency measurement
    - Comparison across configurations
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
    
    def evaluate_query(
        self,
        eval_query: EvaluationQuery,
        results: List[RetrievalResult],
        latency_ms: float,
        total_relevant: int = None
    ) -> QueryResult:
        """Evaluate retrieval results for a single query."""
        
        # Determine which retrieved chunks are relevant
        relevant_indices = []
        for i, result in enumerate(results):
            if eval_query.is_chunk_relevant(result.chunk.text):
                relevant_indices.append(i)
        
        # If we don't know total relevant, assume retrieved relevant = total relevant
        if total_relevant is None:
            total_relevant = max(len(relevant_indices), 1)
        
        # Calculate metrics
        mrr = self._calculate_mrr(relevant_indices)
        precision_at_k = {k: self._calculate_precision_at_k(relevant_indices, k) for k in self.k_values}
        recall_at_k = {k: self._calculate_recall_at_k(relevant_indices, k, total_relevant) for k in self.k_values}
        ndcg_at_k = {k: self._calculate_ndcg_at_k(relevant_indices, k) for k in self.k_values}
        hit_at_k = {k: self._calculate_hit_at_k(relevant_indices, k) for k in self.k_values}
        
        return QueryResult(
            query_id=eval_query.query_id,
            query=eval_query.query,
            retrieved_chunks=results,
            relevant_indices=relevant_indices,
            mrr=mrr,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            hit_at_k=hit_at_k,
            latency_ms=latency_ms
        )
    
    def _calculate_mrr(self, relevant_indices: List[int]) -> float:
        """Mean Reciprocal Rank - position of first relevant result."""
        if not relevant_indices:
            return 0.0
        first_relevant = min(relevant_indices)
        return 1.0 / (first_relevant + 1)
    
    def _calculate_precision_at_k(self, relevant_indices: List[int], k: int) -> float:
        """Precision@K - proportion of relevant docs in top-K."""
        relevant_in_top_k = sum(1 for i in relevant_indices if i < k)
        return relevant_in_top_k / k
    
    def _calculate_recall_at_k(self, relevant_indices: List[int], k: int, total_relevant: int) -> float:
        """Recall@K - proportion of relevant docs found in top-K."""
        if total_relevant == 0:
            return 0.0
        relevant_in_top_k = sum(1 for i in relevant_indices if i < k)
        return relevant_in_top_k / total_relevant
    
    def _calculate_ndcg_at_k(self, relevant_indices: List[int], k: int) -> float:
        """Normalized Discounted Cumulative Gain at K."""
        # Binary relevance: 1 if relevant, 0 otherwise
        relevance = [1 if i in relevant_indices else 0 for i in range(k)]
        
        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        
        # Ideal DCG (all relevant at top)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def _calculate_hit_at_k(self, relevant_indices: List[int], k: int) -> bool:
        """Hit@K - whether any relevant doc is in top-K."""
        return any(i < k for i in relevant_indices)
    
    def evaluate_configuration(
        self,
        config_name: str,
        chunks: List[Chunk],
        retriever: RAGRetriever,
        eval_queries: List[EvaluationQuery],
        chunking_strategy: str,
        retrieval_params: Dict[str, Any]
    ) -> EvaluationReport:
        """Evaluate a complete retrieval configuration."""
        
        # Index chunks
        retriever.index_chunks(chunks)
        
        # Evaluate each query
        query_results = []
        for eq in eval_queries:
            start_time = time.time()
            results = retriever.retrieve(eq.query, top_k=max(self.k_values))
            latency_ms = (time.time() - start_time) * 1000
            
            qr = self.evaluate_query(eq, results, latency_ms)
            query_results.append(qr)
        
        # Create report
        report = EvaluationReport(
            config_name=config_name,
            chunking_strategy=chunking_strategy,
            retrieval_params=retrieval_params,
            query_results=query_results,
            num_chunks=len(chunks),
            avg_chunk_size=sum(len(c.text.split()) for c in chunks) / len(chunks) if chunks else 0
        )
        report.compute_aggregates(self.k_values)
        
        return report


def load_evaluation_queries(filepath: str = "eval_queries.json") -> List[EvaluationQuery]:
    """Load evaluation queries from JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation queries file not found: {filepath}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    queries = []
    for item in data["queries"]:
        queries.append(EvaluationQuery(
            query_id=item["query_id"],
            query=item["query"],
            relevant_keywords=item["relevant_keywords"],
            expected_topics=item.get("expected_topics", []),
            difficulty=item.get("difficulty", "medium")
        ))
    
    return queries


def run_comprehensive_evaluation(
    pages: List[str],
    eval_queries: List[EvaluationQuery],
    output_file: str = "evaluation_results.json"
) -> List[EvaluationReport]:
    """
    Run evaluation across multiple chunking and retrieval configurations.
    
    This is the main function to benchmark different strategies.
    """
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    reports = []
    
    # Define configurations to test
    configurations = [
        # Chunking strategy variations
        {
            "name": "Fixed-200-50 + Hybrid",
            "chunker": get_chunker("fixed", chunk_size=200, overlap=50),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6}
        },
        {
            "name": "Fixed-300-75 + Hybrid",
            "chunker": get_chunker("fixed", chunk_size=300, overlap=75),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6}
        },
        {
            "name": "Sentence-5 + Hybrid",
            "chunker": get_chunker("sentence", max_sentences=5),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6}
        },
        {
            "name": "Semantic + Hybrid",
            "chunker": get_chunker("semantic"),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6}
        },
        {
            "name": "Recursive-300 + Hybrid",
            "chunker": get_chunker("recursive", chunk_size=300),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6}
        },
        {
            "name": "Recursive-500 + Hybrid",
            "chunker": get_chunker("recursive", chunk_size=500),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6}
        },
        # Retrieval approach variations (with best chunking)
        {
            "name": "Recursive-300 + Dense Only",
            "chunker": get_chunker("recursive", chunk_size=300),
            "retrieval_params": {"use_reranking": False, "dense_weight": 1.0, "sparse_weight": 0.0}
        },
        {
            "name": "Recursive-300 + Hybrid (no rerank)",
            "chunker": get_chunker("recursive", chunk_size=300),
            "retrieval_params": {"use_reranking": False, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        {
            "name": "Recursive-300 + Hybrid + Rerank",
            "chunker": get_chunker("recursive", chunk_size=300),
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
    ]
    
    for config in configurations:
        print(f"\nEvaluating: {config['name']}")
        
        # Chunk document
        chunks = config["chunker"].chunk_document(pages)
        print(f"  Chunks created: {len(chunks)}")
        
        # Create retriever
        retriever = RAGRetriever(
            use_reranking=config["retrieval_params"].get("use_reranking", True),
            dense_weight=config["retrieval_params"].get("dense_weight", 0.6),
            sparse_weight=config["retrieval_params"].get("sparse_weight", 0.4)
        )
        
        # Run evaluation
        report = evaluator.evaluate_configuration(
            config_name=config["name"],
            chunks=chunks,
            retriever=retriever,
            eval_queries=eval_queries,
            chunking_strategy=config["chunker"].name,
            retrieval_params=config["retrieval_params"]
        )
        
        reports.append(report)
        
        print(f"  MRR: {report.mean_mrr:.4f}")
        print(f"  Precision@5: {report.mean_precision_at_k.get(5, 0):.4f}")
        print(f"  NDCG@5: {report.mean_ndcg_at_k.get(5, 0):.4f}")
        print(f"  Latency: {report.mean_latency_ms:.2f}ms")
    
    # Save results
    results = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_queries": len(eval_queries),
        "configurations": [r.to_dict() for r in reports]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return reports


def print_comparison_table(reports: List[EvaluationReport]):
    """Print a comparison table of all configurations."""
    print("\n" + "=" * 100)
    print("RETRIEVAL EVALUATION COMPARISON")
    print("=" * 100)
    
    # Header
    print(f"{'Configuration':<35} {'Chunks':>8} {'MRR':>8} {'P@5':>8} {'R@5':>8} {'NDCG@5':>8} {'Hit@5':>8} {'Latency':>10}")
    print("-" * 100)
    
    # Sort by NDCG@5
    sorted_reports = sorted(reports, key=lambda r: r.mean_ndcg_at_k.get(5, 0), reverse=True)
    
    for report in sorted_reports:
        print(f"{report.config_name:<35} "
              f"{report.num_chunks:>8} "
              f"{report.mean_mrr:>8.4f} "
              f"{report.mean_precision_at_k.get(5, 0):>8.4f} "
              f"{report.mean_recall_at_k.get(5, 0):>8.4f} "
              f"{report.mean_ndcg_at_k.get(5, 0):>8.4f} "
              f"{report.hit_rate_at_k.get(5, 0):>8.4f} "
              f"{report.mean_latency_ms:>8.2f}ms")
    
    print("=" * 100)
    
    # Winner
    best = sorted_reports[0]
    print(f"\n🏆 Best configuration: {best.config_name}")
    print(f"   Strategy: {best.chunking_strategy}")
    print(f"   NDCG@5: {best.mean_ndcg_at_k.get(5, 0):.4f}")


