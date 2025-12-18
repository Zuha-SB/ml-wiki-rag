"""
RAG Pipeline - End-to-End Retrieval-Augmented Generation

This module provides the main RAG pipeline that:
1. Loads and processes PDF documents
2. Chunks text using configurable strategies
3. Retrieves relevant context using hybrid search
4. Generates answers using the retrieved context

This is the main interface for the RAG system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pypdf import PdfReader

from .chunking import Chunk, get_chunker
from .retrieval import RAGRetriever, RetrievalResult


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    prompt: str
    chunking_strategy: str
    retrieval_config: Dict[str, Any]


class RAGPipeline:
    """
    End-to-end RAG pipeline for question answering.
    
    Combines document processing, chunking, retrieval, and generation
    into a single cohesive system.
    """
    
    def __init__(
        self,
        chunking_strategy: str = "recursive",
        chunking_params: Optional[Dict[str, Any]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_reranking: bool = True,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunking_strategy: One of "fixed", "sentence", "semantic", "recursive"
            chunking_params: Additional parameters for the chunker
            embedding_model: Sentence transformer model for embeddings
            use_reranking: Whether to use cross-encoder reranking
            dense_weight: Weight for dense retrieval in hybrid fusion
            sparse_weight: Weight for sparse (BM25) retrieval
        """
        self.chunking_strategy = chunking_strategy
        self.chunking_params = chunking_params or {}
        
        # Initialize chunker
        self.chunker = get_chunker(chunking_strategy, **self.chunking_params)
        
        # Initialize retriever
        self.retriever = RAGRetriever(
            embedding_model=embedding_model,
            use_reranking=use_reranking,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        self.chunks: List[Chunk] = []
        self.pages: List[str] = []
        self.is_indexed = False
        
        # Store config for reporting
        self.config = {
            "chunking_strategy": chunking_strategy,
            "chunking_params": chunking_params,
            "embedding_model": embedding_model,
            "use_reranking": use_reranking,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight
        }
    
    def load_pdf(self, pdf_path: str, skip_references: bool = True) -> int:
        """
        Load and extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            skip_references: Whether to skip reference/bibliography sections
        
        Returns:
            Number of pages processed
        """
        import re
        
        reader = PdfReader(pdf_path)
        full_text = ""
        
        # First, extract all text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # Remove References section and everything after it
        if skip_references:
            reference_patterns = [
                "\nReferences\n",
                "\nReferences ",
                "\nREFERENCES\n",
                "\nSee also\n",
                "\nExternal links\n",
                "\nFurther reading\n",
                "\nNotes\n",
                "\nBibliography\n",
            ]
            
            cutoff_index = len(full_text)
            for pattern in reference_patterns:
                idx = full_text.find(pattern)
                if idx != -1 and idx < cutoff_index:
                    cutoff_index = idx
            
            full_text = full_text[:cutoff_index]
        
        # Remove citation markers
        full_text = re.sub(r'\[\d+\]', '', full_text)
        full_text = re.sub(r'\[citation needed\]', '', full_text, flags=re.IGNORECASE)
        full_text = re.sub(r'\[edit\]', '', full_text, flags=re.IGNORECASE)
        full_text = re.sub(r'\[note \d+\]', '', full_text, flags=re.IGNORECASE)
        
        # Split into sections and clean
        self.pages = []
        sections = full_text.split('\n\n')
        
        for section in sections:
            section = section.strip()
            section = re.sub(r'\s+', ' ', section)
            if len(section) > 100:
                self.pages.append(section)
        
        return len(self.pages)
    
    def load_text(self, pages: List[str]) -> None:
        """Load pre-extracted text pages directly."""
        self.pages = pages
    
    def build_index(self) -> Dict[str, Any]:
        """
        Process pages and build the retrieval index.
        
        Returns:
            Statistics about the indexed content
        """
        if not self.pages:
            raise ValueError("No pages loaded. Call load_pdf or load_text first.")
        
        # Chunk all pages
        self.chunks = self.chunker.chunk_document(self.pages)
        
        if not self.chunks:
            raise ValueError("No chunks created from the document.")
        
        # Build retrieval index
        self.retriever.index_chunks(self.chunks)
        self.is_indexed = True
        
        # Calculate statistics
        word_counts = [len(c.text.split()) for c in self.chunks]
        stats = {
            "num_pages": len(self.pages),
            "num_chunks": len(self.chunks),
            "avg_chunk_words": sum(word_counts) / len(word_counts),
            "min_chunk_words": min(word_counts),
            "max_chunk_words": max(word_counts),
            "chunking_strategy": self.chunking_strategy
        }
        
        return stats
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve
        
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index first.")
        
        return self.retriever.retrieve(query, top_k=top_k)
    
    def format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved results as context for generation."""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Passage {i}]\n{result.chunk.text}")
        return "\n\n".join(context_parts)
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the language model."""
        prompt = f"""You are an expert explaining machine learning concepts. Use the reference material below to answer the question.

Important: Answer in your own words by synthesizing the information. Do NOT copy text verbatim from the context.
If the answer cannot be found in the context, say "I cannot find this information in the provided context."

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        return_prompt: bool = False
    ) -> RAGResponse:
        """
        Full RAG query: retrieve context and prepare for generation.
        
        Note: This returns the prompt and context. Actual generation
        depends on your LLM setup (OpenAI, local model, etc.)
        
        Args:
            query: The user's question
            top_k: Number of context chunks to retrieve
            return_prompt: Whether to include the full prompt in response
        
        Returns:
            RAGResponse with retrieved context and prepared prompt
        """
        # Retrieve
        results = self.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.format_context(results)
        
        # Build prompt
        prompt = self.build_prompt(query, context)
        
        # Format retrieved chunks for response
        retrieved_info = []
        for r in results:
            retrieved_info.append({
                "text": r.chunk.text,
                "chunk_id": r.chunk.chunk_id,
                "page": r.chunk.source_page,
                "rank": r.rank,
                "score": round(r.score, 4),
                "dense_score": round(r.dense_score, 4) if r.dense_score else None,
                "sparse_score": round(r.sparse_score, 4) if r.sparse_score else None,
                "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None
            })
        
        return RAGResponse(
            query=query,
            answer="",  # To be filled by LLM
            retrieved_chunks=retrieved_info,
            prompt=prompt if return_prompt else "",
            chunking_strategy=self.chunking_strategy,
            retrieval_config=self.config
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        if not self.is_indexed:
            return {"status": "not_indexed"}
        
        word_counts = [len(c.text.split()) for c in self.chunks]
        return {
            "status": "indexed",
            "num_pages": len(self.pages),
            "num_chunks": len(self.chunks),
            "avg_chunk_words": round(sum(word_counts) / len(word_counts), 1),
            "config": self.config
        }


def create_pipeline_from_pdf(
    pdf_path: str,
    chunking_strategy: str = "recursive",
    **kwargs
) -> RAGPipeline:
    """
    Convenience function to create a pipeline from a PDF file.
    
    Args:
        pdf_path: Path to the PDF
        chunking_strategy: Chunking strategy to use
        **kwargs: Additional arguments for RAGPipeline
    
    Returns:
        Initialized and indexed RAGPipeline
    """
    pipeline = RAGPipeline(chunking_strategy=chunking_strategy, **kwargs)
    pipeline.load_pdf(pdf_path)
    pipeline.build_index()
    return pipeline


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--pdf", "-p", default="Machine_learning.pdf", help="PDF file path")
    parser.add_argument("--query", "-q", help="Query to run")
    parser.add_argument("--strategy", "-s", default="recursive", 
                       choices=["fixed", "sentence", "semantic", "recursive"],
                       help="Chunking strategy")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    
    args = parser.parse_args()
    
    print(f"Loading PDF: {args.pdf}")
    pipeline = RAGPipeline(
        chunking_strategy=args.strategy,
        use_reranking=not args.no_rerank
    )
    
    num_pages = pipeline.load_pdf(args.pdf)
    print(f"Loaded {num_pages} pages")
    
    stats = pipeline.build_index()
    print(f"Created {stats['num_chunks']} chunks (avg {stats['avg_chunk_words']:.1f} words)")
    
    if args.query:
        query = args.query
    else:
        query = input("\nEnter your question: ")
    
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    response = pipeline.query(query, top_k=args.top_k, return_prompt=True)
    
    print("\nRetrieved chunks:")
    for chunk in response.retrieved_chunks:
        print(f"\n[Rank {chunk['rank']}] Score: {chunk['score']}")
        if chunk['dense_score']:
            print(f"  Dense: {chunk['dense_score']}, Sparse: {chunk['sparse_score']}, Rerank: {chunk['rerank_score']}")
        print(f"  {chunk['text'][:200]}...")
    
    print("\n" + "=" * 50)
    print("PROMPT FOR LLM:")
    print("=" * 50)
    print(response.prompt)

