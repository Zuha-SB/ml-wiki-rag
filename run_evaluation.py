#!/usr/bin/env python3
"""
Run comprehensive evaluation of the RAG pipeline.

This script evaluates multiple chunking and retrieval configurations
and outputs a comparison table with all metrics.

Usage:
    python run_evaluation.py
    python run_evaluation.py --output results.json
"""

import argparse
import json
import time
from pypdf import PdfReader

from src.chunking import get_chunker
from src.retrieval import RAGRetriever
from src.evaluation import (
    load_evaluation_queries,
    RetrievalEvaluator,
    EvaluationReport,
    print_comparison_table
)


def load_pdf(pdf_path: str = "Machine_learning.pdf") -> list:
    """Load and extract text from PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            text_lower = text.strip().lower()
            if not text_lower.startswith("references") and not text_lower.startswith("external links"):
                pages.append(text.replace('\n', ' '))
    
    return pages


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--pdf", default="Machine_learning.pdf", help="PDF file path")
    parser.add_argument("--queries", default="data/eval_queries.json", help="Evaluation queries JSON")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RAG PIPELINE EVALUATION")
    print("=" * 70)
    
    # Load PDF
    print(f"\n📄 Loading PDF: {args.pdf}")
    pages = load_pdf(args.pdf)
    print(f"   Loaded {len(pages)} pages")
    
    # Load evaluation queries
    print(f"\n📝 Loading evaluation queries: {args.queries}")
    try:
        eval_queries = load_evaluation_queries(args.queries)
        print(f"   Loaded {len(eval_queries)} evaluation queries")
    except FileNotFoundError:
        print(f"   ERROR: File not found: {args.queries}")
        return
    
    # Define configurations to test
    configurations = [
        # Chunking variations with hybrid retrieval
        {
            "name": "Fixed-200 + Hybrid + Rerank",
            "chunker_strategy": "fixed",
            "chunker_params": {"chunk_size": 200, "overlap": 50},
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        {
            "name": "Fixed-300 + Hybrid + Rerank",
            "chunker_strategy": "fixed",
            "chunker_params": {"chunk_size": 300, "overlap": 75},
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        {
            "name": "Sentence-5 + Hybrid + Rerank",
            "chunker_strategy": "sentence",
            "chunker_params": {"max_sentences": 5},
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        {
            "name": "Semantic + Hybrid + Rerank",
            "chunker_strategy": "semantic",
            "chunker_params": {},
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        {
            "name": "Recursive-300 + Hybrid + Rerank",
            "chunker_strategy": "recursive",
            "chunker_params": {"chunk_size": 300},
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        {
            "name": "Recursive-500 + Hybrid + Rerank",
            "chunker_strategy": "recursive",
            "chunker_params": {"chunk_size": 500},
            "retrieval_params": {"use_reranking": True, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
        # Retrieval variations with best chunking
        {
            "name": "Recursive-300 + Dense Only",
            "chunker_strategy": "recursive",
            "chunker_params": {"chunk_size": 300},
            "retrieval_params": {"use_reranking": False, "dense_weight": 1.0, "sparse_weight": 0.0}
        },
        {
            "name": "Recursive-300 + Hybrid (no rerank)",
            "chunker_strategy": "recursive",
            "chunker_params": {"chunk_size": 300},
            "retrieval_params": {"use_reranking": False, "dense_weight": 0.6, "sparse_weight": 0.4}
        },
    ]
    
    # Run evaluation
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    reports = []
    
    print(f"\n🧪 Running evaluation on {len(configurations)} configurations...")
    print("-" * 70)
    
    for i, config in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] {config['name']}")
        start_time = time.time()
        
        # Create chunker and chunks
        chunker = get_chunker(config["chunker_strategy"], **config["chunker_params"])
        chunks = chunker.chunk_document(pages)
        print(f"    📦 Created {len(chunks)} chunks")
        
        # Create retriever
        retriever = RAGRetriever(
            use_reranking=config["retrieval_params"]["use_reranking"],
            dense_weight=config["retrieval_params"]["dense_weight"],
            sparse_weight=config["retrieval_params"]["sparse_weight"]
        )
        
        # Run evaluation
        report = evaluator.evaluate_configuration(
            config_name=config["name"],
            chunks=chunks,
            retriever=retriever,
            eval_queries=eval_queries,
            chunking_strategy=config["chunker_strategy"],
            retrieval_params=config["retrieval_params"]
        )
        
        elapsed = time.time() - start_time
        print(f"    📊 MRR: {report.mean_mrr:.4f} | P@5: {report.mean_precision_at_k.get(5, 0):.4f} | "
              f"NDCG@5: {report.mean_ndcg_at_k.get(5, 0):.4f} | Time: {elapsed:.1f}s")
        
        reports.append(report)
    
    # Print comparison table
    print_comparison_table(reports)
    
    # Save results
    results = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_file": args.pdf,
        "num_pages": len(pages),
        "num_queries": len(eval_queries),
        "configurations": [r.to_dict() for r in reports]
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    
    # Summary
    best_report = max(reports, key=lambda r: r.mean_ndcg_at_k.get(5, 0))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"📊 Best configuration: {best_report.config_name}")
    print(f"   - NDCG@5: {best_report.mean_ndcg_at_k.get(5, 0):.4f}")
    print(f"   - MRR: {best_report.mean_mrr:.4f}")
    print(f"   - Precision@5: {best_report.mean_precision_at_k.get(5, 0):.4f}")
    print(f"   - Latency: {best_report.mean_latency_ms:.2f}ms")


if __name__ == "__main__":
    main()


