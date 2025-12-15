"""
RAG Web Application

A modern web interface for the RAG pipeline that showcases:
- Multiple chunking strategies with live comparison
- Hybrid retrieval with detailed scoring breakdown
- Evaluation metrics and benchmarking
"""

import json
import os
from flask import Flask, request, jsonify, render_template_string
from pypdf import PdfReader
import time

from chunking import get_chunker, compare_chunking_strategies, Chunk
from retrieval import RAGRetriever
from rag_pipeline import RAGPipeline
from evaluation import load_evaluation_queries, RetrievalEvaluator, EvaluationQuery

# Import generator
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = Flask(__name__)

# Global state
PDF_PATH = "Machine_learning.pdf"
pipelines = {}  # Cache pipelines by strategy
pages = []  # Cached PDF pages
generator = None  # LLM generator


def load_pdf_pages():
    """Load PDF pages once at startup with proper cleaning."""
    global pages
    if pages:
        return pages
    
    reader = PdfReader(PDF_PATH)
    full_text = ""
    
    # First, extract all text
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    
    # Remove References section and everything after it
    # Common patterns for references section in Wikipedia PDFs
    reference_patterns = [
        "\nReferences\n",
        "\nReferences ",
        "\nREFERENCES\n",
        "\nSee also\n",  # Often comes before references
        "\nExternal links\n",
        "\nFurther reading\n",
        "\nNotes\n",
        "\nBibliography\n",
    ]
    
    # Find the earliest occurrence of any reference pattern
    cutoff_index = len(full_text)
    for pattern in reference_patterns:
        idx = full_text.find(pattern)
        if idx != -1 and idx < cutoff_index:
            cutoff_index = idx
    
    # Keep only content before references
    clean_text = full_text[:cutoff_index]
    
    # Also remove citation markers like [1], [2], [citation needed], etc.
    import re
    clean_text = re.sub(r'\[\d+\]', '', clean_text)  # Remove [1], [2], etc.
    clean_text = re.sub(r'\[citation needed\]', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\[edit\]', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\[note \d+\]', '', clean_text, flags=re.IGNORECASE)
    
    # Split into logical pages/sections (by double newlines or page breaks)
    # and clean up whitespace
    sections = clean_text.split('\n\n')
    
    for section in sections:
        # Clean up the section
        section = section.strip()
        section = re.sub(r'\s+', ' ', section)  # Normalize whitespace
        
        # Skip very short sections (likely headers or artifacts)
        if len(section) > 100:
            pages.append(section)
    
    print(f"Cleaned text: {len(pages)} sections, ~{sum(len(p) for p in pages)} characters")
    
    # Save cleaned text to file for inspection
    with open("cleaned_text.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLEANED & PREPROCESSED TEXT FROM PDF\n")
        f.write("=" * 80 + "\n\n")
        for i, section in enumerate(pages, 1):
            f.write(f"--- Section {i} ({len(section.split())} words) ---\n")
            f.write(section + "\n\n")
    print(f"Saved cleaned text to: cleaned_text.txt")
    
    return pages


def get_pipeline(strategy: str = "recursive", use_reranking: bool = True) -> RAGPipeline:
    """Get or create a pipeline for a given strategy."""
    cache_key = f"{strategy}_{use_reranking}"
    
    if cache_key not in pipelines:
        pipeline = RAGPipeline(
            chunking_strategy=strategy,
            use_reranking=use_reranking
        )
        pipeline.load_text(load_pdf_pages())
        pipeline.build_index()
        pipelines[cache_key] = pipeline
        
        # Save chunks to file for inspection
        chunks_file = f"chunks_{strategy}.txt"
        with open(chunks_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"CHUNKS CREATED WITH '{strategy.upper()}' STRATEGY\n")
            f.write(f"Total chunks: {len(pipeline.chunks)}\n")
            f.write("=" * 80 + "\n\n")
            for i, chunk in enumerate(pipeline.chunks, 1):
                word_count = len(chunk.text.split())
                f.write(f"--- Chunk {i} (Page {chunk.source_page}, {word_count} words) ---\n")
                f.write(chunk.text + "\n\n")
        print(f"Saved chunks to: {chunks_file}")
    
    return pipelines[cache_key]


def get_generator():
    """Get or create the text generator (using Mistral-7B-Instruct via llama.cpp)."""
    global generator
    if generator is None:
        print("Loading Mistral-7B-Instruct model...")
        print("First run will download the model (~4.4GB). Please wait...")
        
        # Download quantized GGUF model from HuggingFace
        model_path = hf_hub_download(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        )
        
        print(f"Model downloaded to: {model_path}")
        print("Loading model into memory...")
        
        # Load the model with llama.cpp
        # n_ctx: context window, n_threads: CPU threads
        generator = Llama(
            model_path=model_path,
            n_ctx=4096,  # Context window
            n_threads=8,  # Adjust based on your CPU
            n_gpu_layers=-1,  # Set >0 if you have GPU with CUDA
            verbose=False
        )
        
        print("Mistral-7B-Instruct ready!")
    return generator


def generate_response(prompt: str, max_tokens: int = 200) -> str:
    """Generate a response using Mistral-7B-Instruct with RAG context."""
    llm = get_generator()
    
    # Extract context and question from the prompt
    context_start = prompt.find("Context:\n")
    question_start = prompt.find("\n\nQuestion:")
    
    if context_start >= 0 and question_start >= 0:
        context = prompt[context_start + 9:question_start].strip()
        question_end = prompt.find("\n\nAnswer:")
        question = prompt[question_start + 11:question_end].strip() if question_end > 0 else prompt[question_start + 11:].strip()
    else:
        context = ""
        question = prompt
    
    # Mistral instruction format with [INST] tags
    # Craft a prompt that encourages synthesis, not copying
    mistral_prompt = f"""<s>[INST] You are a knowledgeable assistant helping explain machine learning concepts.

Based on the following reference information, answer the user's question. 
Synthesize the information into a clear, well-structured response. 
Explain concepts in your own words - do not copy text verbatim.
If the information isn't in the context, say so.

Reference Information:
{context}

User Question: {question}

Provide a helpful, synthesized answer: [/INST]"""
    
    try:
        # Generate response
        output = llm(
            mistral_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "[INST]"],  # Stop tokens
            echo=False  # Don't include prompt in output
        )
        
        response = output["choices"][0]["text"].strip()
        
        return response if response else "I could not generate a response based on the context."
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_response_no_rag(question: str, max_tokens: int = 200) -> str:
    """Generate a response using Mistral-7B-Instruct WITHOUT any RAG context."""
    llm = get_generator()
    
    # Simple prompt without any retrieved context
    mistral_prompt = f"""<s>[INST] You are a knowledgeable assistant helping explain machine learning concepts.

Answer the following question about machine learning. Provide a clear, informative response based on your knowledge.

Question: {question}

Answer: [/INST]"""
    
    try:
        output = llm(
            mistral_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "[INST]"],
            echo=False
        )
        
        response = output["choices"][0]["text"].strip()
        return response if response else "I could not generate a response."
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Load pages at startup
print("Loading PDF...")
load_pdf_pages()
print(f"Loaded {len(pages)} pages")

# Pre-load default pipeline
print("Building default index...")
get_pipeline("recursive", True)

# Pre-load generator
print("Loading generator...")
get_generator()
print("Ready!")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Wiki RAG | Retrieval-Augmented Generation</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --bg-card: #15151f;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #2d2d3a;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #3b82f6;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Background pattern */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(168, 85, 247, 0.05) 0%, transparent 70%);
            pointer-events: none;
            z-index: -1;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header */
        header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem 0;
        }
        
        .logo {
            display: inline-flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .logo-icon {
            width: 56px;
            height: 56px;
            background: var(--accent-gradient);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.75rem;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .tagline {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            background: var(--bg-secondary);
            padding: 0.5rem;
            border-radius: 16px;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }
        
        .tab {
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-radius: 12px;
            transition: all 0.2s ease;
        }
        
        .tab:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }
        
        .tab.active {
            background: var(--accent-gradient);
            color: white;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }
        
        /* Panels */
        .panel {
            display: none;
        }
        
        .panel.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Forms */
        .form-group {
            margin-bottom: 1.25rem;
        }
        
        label {
            display: block;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 0.875rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 1rem;
            transition: all 0.2s ease;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        /* Grid */
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }
        
        @media (max-width: 768px) {
            .grid-2, .grid-3 {
                grid-template-columns: 1fr;
            }
        }
        
        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.875rem 1.5rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: white;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: var(--bg-tertiary);
            box-shadow: none;
        }
        
        .btn-secondary:hover {
            background: var(--bg-secondary);
            box-shadow: none;
        }
        
        /* Results */
        .result-chunk {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        
        .result-chunk:hover {
            border-color: var(--accent-primary);
        }
        
        .chunk-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
        }
        
        .rank-badge {
            width: 32px;
            height: 32px;
            background: var(--accent-gradient);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .score-pills {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .pill {
            padding: 0.25rem 0.75rem;
            background: var(--bg-tertiary);
            border-radius: 20px;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }
        
        .pill.primary {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
        }
        
        .chunk-text {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.7;
        }
        
        .chunk-meta {
            margin-top: 0.75rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        
        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.75rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        
        /* Metrics table */
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        
        .metrics-table th, .metrics-table td {
            padding: 0.875rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .metrics-table th {
            color: var(--text-secondary);
            font-weight: 500;
            background: var(--bg-secondary);
        }
        
        .metrics-table tr:hover td {
            background: var(--bg-secondary);
        }
        
        .metrics-table .best {
            color: var(--success);
            font-weight: 600;
        }
        
        /* Loading spinner */
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid var(--bg-tertiary);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(10, 10, 15, 0.8);
            backdrop-filter: blur(4px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 1rem;
        }
        
        .loading-overlay.active {
            display: flex;
        }
        
        .loading-text {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        
        /* Answer section */
        .answer-content {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 1.25rem;
            font-size: 1.05rem;
            line-height: 1.8;
            color: var(--text-primary);
        }
        
        .answer-content.no-rag {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .answer-meta {
            margin-top: 0.75rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
        
        .answer-label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .answer-label.rag {
            color: var(--accent-primary);
        }
        
        .answer-label.no-rag {
            color: var(--danger);
        }
        
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        
        .comparison-item {
            display: flex;
            flex-direction: column;
        }
        
        @media (max-width: 900px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* Prompt preview */
        .prompt-preview {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        
        /* Toggle */
        .toggle-group {
            display: flex;
            gap: 0.5rem;
        }
        
        .toggle-btn {
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .toggle-btn.active {
            background: var(--accent-primary);
            border-color: var(--accent-primary);
            color: white;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-muted);
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        /* Checkbox */
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: auto;
            accent-color: var(--accent-primary);
        }
        
        .checkbox-group label {
            margin: 0;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loading">
        <div class="spinner"></div>
        <div class="loading-text">Processing your request...</div>
    </div>
    
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">🔍</div>
                <h1>ML Wiki RAG</h1>
            </div>
            <p class="tagline">Retrieval-Augmented Generation for Machine Learning Knowledge</p>
        </header>
        
        <div class="tabs">
            <button class="tab active" onclick="showPanel('query')">💬 Query</button>
            <button class="tab" onclick="showPanel('chunking')">📦 Chunking</button>
            <button class="tab" onclick="showPanel('evaluation')">📊 Evaluation</button>
            <button class="tab" onclick="showPanel('about')">ℹ️ About</button>
        </div>
        
        <!-- Query Panel -->
        <div class="panel active" id="panel-query">
            <div class="grid-2">
                <div>
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">⚙️ Configuration</span>
                        </div>
                        
                        <div class="form-group">
                            <label>Chunking Strategy</label>
                            <select id="chunk-strategy">
                                <option value="recursive">Recursive (recommended)</option>
                                <option value="semantic">Semantic</option>
                                <option value="sentence">Sentence-based</option>
                                <option value="fixed">Fixed-size</option>
                            </select>
                        </div>
                        
                        <div class="grid-2">
                            <div class="form-group">
                                <label>Top-K Results</label>
                                <select id="top-k">
                                    <option value="3">3</option>
                                    <option value="5" selected>5</option>
                                    <option value="10">10</option>
                                </select>
                            </div>
                            <div class="form-group" style="display: flex; flex-direction: column; gap: 0.5rem; justify-content: flex-end;">
                                <div class="checkbox-group">
                                    <input type="checkbox" id="use-rerank" checked>
                                    <label for="use-rerank">Use Re-ranking</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" id="generate-answer" checked>
                                    <label for="generate-answer">Generate Answer</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" id="compare-no-rag">
                                    <label for="compare-no-rag">Compare with No-RAG</label>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">❓ Your Question</span>
                        </div>
                        <div class="form-group">
                            <textarea id="query-input" placeholder="Ask anything about machine learning..."></textarea>
                        </div>
                        <button class="btn" onclick="submitQuery()">
                            🔍 Search & Retrieve
                        </button>
                    </div>
                    
                    <div class="card" id="answer-card" style="display: none;">
                        <div class="card-header">
                            <span class="card-title">💡 Generated Answer</span>
                        </div>
                        <div id="answers-container"></div>
                    </div>
                    
                    <div class="card" id="prompt-card" style="display: none;">
                        <div class="card-header">
                            <span class="card-title">📝 Full Prompt (for reference)</span>
                            <button class="btn btn-secondary" onclick="copyPrompt()" style="padding: 0.5rem 1rem; font-size: 0.85rem;">
                                📋 Copy
                            </button>
                        </div>
                        <div class="prompt-preview" id="prompt-preview"></div>
                    </div>
                </div>
                
                <div>
                    <div class="card" id="stats-card" style="display: none;">
                        <div class="card-header">
                            <span class="card-title">📈 Pipeline Stats</span>
                        </div>
                        <div class="stats-grid" id="pipeline-stats"></div>
                    </div>
                    
                    <div id="results-container">
                        <div class="empty-state">
                            <div class="empty-state-icon">🔎</div>
                            <p>Enter a question to retrieve relevant passages</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Chunking Panel -->
        <div class="panel" id="panel-chunking">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📊 Chunking Strategy Comparison</span>
                    <button class="btn" onclick="runChunkingComparison()">Run Comparison</button>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Compare different chunking strategies on the same document to understand their trade-offs.
                </p>
                <div id="chunking-results">
                    <div class="empty-state">
                        <div class="empty-state-icon">📦</div>
                        <p>Click "Run Comparison" to analyze chunking strategies</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span class="card-title">💡 Strategy Guide</span>
                </div>
                <div class="grid-2">
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-primary);">Fixed-Size</h4>
                        <p class="chunk-text">Splits text into chunks of approximately equal size (by words or characters) with overlap. Simple and predictable, but may split mid-sentence.</p>
                    </div>
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-primary);">Sentence-Based</h4>
                        <p class="chunk-text">Groups complete sentences until reaching a target size. Preserves sentence boundaries for more coherent chunks, but variable sizes.</p>
                    </div>
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-primary);">Semantic</h4>
                        <p class="chunk-text">Identifies natural text boundaries like paragraphs and sections. Best for preserving topical coherence, but may create uneven chunks.</p>
                    </div>
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-primary);">Recursive</h4>
                        <p class="chunk-text">Hierarchically splits using multiple separators (paragraphs → sentences → words). Adapts to content structure for optimal balance.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Evaluation Panel -->
        <div class="panel" id="panel-evaluation">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🧪 Retrieval Evaluation</span>
                    <button class="btn" onclick="runEvaluation()">Run Full Evaluation</button>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Evaluate retrieval quality across multiple configurations using standard IR metrics: MRR, Precision@K, Recall@K, NDCG@K.
                </p>
                <div id="eval-results">
                    <div class="empty-state">
                        <div class="empty-state-icon">📊</div>
                        <p>Click "Run Full Evaluation" to benchmark all configurations</p>
                        <p style="font-size: 0.85rem; margin-top: 0.5rem;">This may take a few minutes...</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📚 Metrics Explained</span>
                </div>
                <div class="grid-2">
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--success);">MRR (Mean Reciprocal Rank)</h4>
                        <p class="chunk-text">Average of 1/rank of the first relevant result. Higher is better (max 1.0). Measures how quickly you find relevant content.</p>
                    </div>
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--info);">Precision@K</h4>
                        <p class="chunk-text">Proportion of retrieved documents in top-K that are relevant. Measures accuracy of the top results.</p>
                    </div>
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--warning);">Recall@K</h4>
                        <p class="chunk-text">Proportion of all relevant documents found in top-K. Measures coverage of relevant information.</p>
                    </div>
                    <div class="result-chunk">
                        <h4 style="margin-bottom: 0.5rem; color: var(--accent-secondary);">NDCG@K</h4>
                        <p class="chunk-text">Normalized Discounted Cumulative Gain. Accounts for position in ranking—relevant docs at top positions are weighted more heavily.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- About Panel -->
        <div class="panel" id="panel-about">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🎯 About This RAG System</span>
                </div>
                <div style="color: var(--text-secondary); line-height: 1.8;">
                    <p>This is a comprehensive Retrieval-Augmented Generation (RAG) pipeline designed for the Machine Learning Wikipedia dataset. It demonstrates:</p>
                    <br>
                    <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">📦 Chunking Strategies</h4>
                    <ul style="margin-left: 1.5rem; margin-bottom: 1rem;">
                        <li><strong>Fixed-Size:</strong> Word/character-based with overlap</li>
                        <li><strong>Sentence-Based:</strong> Respects sentence boundaries</li>
                        <li><strong>Semantic:</strong> Paragraph/section-aware</li>
                        <li><strong>Recursive:</strong> Hierarchical with multiple separators</li>
                    </ul>
                    
                    <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">🔍 Retrieval Approach</h4>
                    <ul style="margin-left: 1.5rem; margin-bottom: 1rem;">
                        <li><strong>Dense Retrieval:</strong> Sentence transformer embeddings + FAISS</li>
                        <li><strong>Sparse Retrieval:</strong> BM25 lexical matching</li>
                        <li><strong>Hybrid Fusion:</strong> Reciprocal Rank Fusion (RRF)</li>
                        <li><strong>Re-ranking:</strong> Cross-encoder for improved precision</li>
                    </ul>
                    
                    <h4 style="color: var(--text-primary); margin-bottom: 0.5rem;">📊 Evaluation Metrics</h4>
                    <ul style="margin-left: 1.5rem;">
                        <li>Mean Reciprocal Rank (MRR)</li>
                        <li>Precision@K, Recall@K</li>
                        <li>Normalized Discounted Cumulative Gain (NDCG@K)</li>
                        <li>Hit Rate@K</li>
                    </ul>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🛠️ Technical Stack</span>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">🤗</div>
                        <div class="stat-label">Sentence Transformers</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">⚡</div>
                        <div class="stat-label">FAISS</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">🔤</div>
                        <div class="stat-label">BM25</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">🎯</div>
                        <div class="stat-label">Cross-Encoder</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">🧠</div>
                        <div class="stat-label">Mistral-7B</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">🌐</div>
                        <div class="stat-label">Flask</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showPanel(panelId) {
            // Hide all panels
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            
            // Show selected panel
            document.getElementById('panel-' + panelId).classList.add('active');
            event.target.classList.add('active');
        }
        
        function showLoading(show) {
            document.getElementById('loading').classList.toggle('active', show);
        }
        
        async function submitQuery() {
            const query = document.getElementById('query-input').value.trim();
            if (!query) return;
            
            const strategy = document.getElementById('chunk-strategy').value;
            const topK = parseInt(document.getElementById('top-k').value);
            const useRerank = document.getElementById('use-rerank').checked;
            const generateAnswer = document.getElementById('generate-answer').checked;
            const compareNoRag = document.getElementById('compare-no-rag').checked;
            
            showLoading(true);
            let loadingText = 'Retrieving passages...';
            if (generateAnswer && compareNoRag) {
                loadingText = 'Retrieving and generating both RAG & No-RAG responses...';
            } else if (generateAnswer) {
                loadingText = 'Retrieving and generating response...';
            }
            document.querySelector('#loading .loading-text').textContent = loadingText;
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        strategy: strategy,
                        top_k: topK,
                        use_reranking: useRerank,
                        generate: generateAnswer,
                        compare_no_rag: compareNoRag
                    })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('results-container').innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">❌</div>
                        <p>Error processing request: ${error.message}</p>
                    </div>
                `;
            } finally {
                showLoading(false);
                document.querySelector('#loading .loading-text').textContent = 'Processing your request...';
            }
        }
        
        function displayResults(data) {
            // Show stats
            const statsCard = document.getElementById('stats-card');
            statsCard.style.display = 'block';
            
            const stats = data.stats;
            document.getElementById('pipeline-stats').innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.num_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.avg_chunk_words}</div>
                    <div class="stat-label">Avg Words</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.retrieval_ms}ms</div>
                    <div class="stat-label">Retrieval</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.generation_ms}ms</div>
                    <div class="stat-label">Generation</div>
                </div>
            `;
            
            // Show generated answer(s)
            if (data.answer) {
                document.getElementById('answer-card').style.display = 'block';
                
                let answersHtml = '';
                
                if (data.no_rag_answer) {
                    // Show side-by-side comparison
                    answersHtml = `
                        <div class="comparison-grid">
                            <div class="comparison-item">
                                <div class="answer-label rag">✅ With RAG (Retrieved Context)</div>
                                <div class="answer-content">${escapeHtml(data.answer)}</div>
                                <div class="answer-meta">Generated in ${data.generation_ms}ms</div>
                            </div>
                            <div class="comparison-item">
                                <div class="answer-label no-rag">❌ Without RAG (No Context)</div>
                                <div class="answer-content no-rag">${escapeHtml(data.no_rag_answer)}</div>
                                <div class="answer-meta">Generated in ${data.no_rag_generation_ms}ms</div>
                            </div>
                        </div>
                    `;
                } else {
                    // Show single answer
                    answersHtml = `
                        <div class="answer-content">${escapeHtml(data.answer)}</div>
                        <div class="answer-meta">Generated in ${data.generation_ms}ms using Mistral-7B-Instruct</div>
                    `;
                }
                
                document.getElementById('answers-container').innerHTML = answersHtml;
            } else {
                document.getElementById('answer-card').style.display = 'none';
            }
            
            // Show results
            let resultsHtml = '<h3 style="margin-bottom: 1rem;">🎯 Retrieved Passages</h3>';
            
            for (const chunk of data.results) {
                const denseScore = chunk.dense_score !== null ? `Dense: ${chunk.dense_score}` : '';
                const sparseScore = chunk.sparse_score !== null ? `BM25: ${chunk.sparse_score}` : '';
                const rerankScore = chunk.rerank_score !== null ? `Rerank: ${chunk.rerank_score}` : '';
                
                resultsHtml += `
                    <div class="result-chunk">
                        <div class="chunk-header">
                            <div class="rank-badge">${chunk.rank}</div>
                            <div class="score-pills">
                                <span class="pill primary">Score: ${chunk.score}</span>
                                ${denseScore ? `<span class="pill">${denseScore}</span>` : ''}
                                ${sparseScore ? `<span class="pill">${sparseScore}</span>` : ''}
                                ${rerankScore ? `<span class="pill">${rerankScore}</span>` : ''}
                            </div>
                        </div>
                        <div class="chunk-text">${escapeHtml(chunk.text)}</div>
                        <div class="chunk-meta">Page ${chunk.page || 'N/A'} • ${chunk.text.split(' ').length} words</div>
                    </div>
                `;
            }
            
            document.getElementById('results-container').innerHTML = resultsHtml;
            
            // Show prompt
            document.getElementById('prompt-card').style.display = 'block';
            document.getElementById('prompt-preview').textContent = data.prompt;
        }
        
        function copyPrompt() {
            const prompt = document.getElementById('prompt-preview').textContent;
            navigator.clipboard.writeText(prompt);
            alert('Prompt copied to clipboard!');
        }
        
        async function runChunkingComparison() {
            showLoading(true);
            
            try {
                const response = await fetch('/api/chunking-comparison');
                const data = await response.json();
                
                let html = `
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Chunks</th>
                                <th>Avg Words</th>
                                <th>Min Words</th>
                                <th>Max Words</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                for (const [name, stats] of Object.entries(data)) {
                    html += `
                        <tr>
                            <td><strong>${name}</strong></td>
                            <td>${stats.num_chunks}</td>
                            <td>${stats.avg_words.toFixed(1)}</td>
                            <td>${stats.min_words}</td>
                            <td>${stats.max_words}</td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                document.getElementById('chunking-results').innerHTML = html;
                
            } catch (error) {
                console.error('Error:', error);
            } finally {
                showLoading(false);
            }
        }
        
        async function runEvaluation() {
            showLoading(true);
            document.querySelector('#loading .loading-text').textContent = 'Running evaluation (this may take a few minutes)...';
            
            try {
                const response = await fetch('/api/evaluate');
                const data = await response.json();
                
                let html = `
                    <p style="color: var(--success); margin-bottom: 1rem;">✅ Evaluated ${data.num_queries} queries across ${data.configurations.length} configurations</p>
                    <div style="overflow-x: auto;">
                        <table class="metrics-table">
                            <thead>
                                <tr>
                                    <th>Configuration</th>
                                    <th>Chunks</th>
                                    <th>MRR</th>
                                    <th>P@5</th>
                                    <th>R@5</th>
                                    <th>NDCG@5</th>
                                    <th>Hit@5</th>
                                    <th>Latency</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                // Find best values
                const bestMRR = Math.max(...data.configurations.map(c => c.metrics.mean_mrr));
                const bestNDCG = Math.max(...data.configurations.map(c => c.metrics['ndcg@k']['5'] || 0));
                
                for (const config of data.configurations) {
                    const m = config.metrics;
                    const isBestMRR = m.mean_mrr === bestMRR;
                    const isBestNDCG = (m['ndcg@k']['5'] || 0) === bestNDCG;
                    
                    html += `
                        <tr>
                            <td><strong>${config.config_name}</strong></td>
                            <td>${config.num_chunks}</td>
                            <td class="${isBestMRR ? 'best' : ''}">${m.mean_mrr.toFixed(4)}</td>
                            <td>${(m['precision@k']['5'] || 0).toFixed(4)}</td>
                            <td>${(m['recall@k']['5'] || 0).toFixed(4)}</td>
                            <td class="${isBestNDCG ? 'best' : ''}">${(m['ndcg@k']['5'] || 0).toFixed(4)}</td>
                            <td>${(m['hit_rate@k']['5'] || 0).toFixed(4)}</td>
                            <td>${m.mean_latency_ms.toFixed(1)}ms</td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table></div>';
                document.getElementById('eval-results').innerHTML = html;
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('eval-results').innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">❌</div>
                        <p>Error running evaluation: ${error.message}</p>
                    </div>
                `;
            } finally {
                showLoading(false);
                document.querySelector('#loading .loading-text').textContent = 'Processing your request...';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Allow Enter to submit
        document.getElementById('query-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuery();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    query = data.get('query', '')
    strategy = data.get('strategy', 'recursive')
    top_k = data.get('top_k', 5)
    use_reranking = data.get('use_reranking', True)
    generate = data.get('generate', True)  # Whether to generate response
    compare_no_rag = data.get('compare_no_rag', False)  # Compare with no-RAG response
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Get or create pipeline
    pipeline = get_pipeline(strategy, use_reranking)
    
    # Query
    start_time = time.time()
    response = pipeline.query(query, top_k=top_k, return_prompt=True)
    retrieval_time = time.time() - start_time
    
    # Generate RAG response if requested
    generated_response = ""
    generation_time = 0
    if generate:
        gen_start = time.time()
        generated_response = generate_response(response.prompt)
        generation_time = time.time() - gen_start
    
    # Generate No-RAG response for comparison if requested
    no_rag_response = ""
    no_rag_time = 0
    if compare_no_rag and generate:
        no_rag_start = time.time()
        no_rag_response = generate_response_no_rag(query)
        no_rag_time = time.time() - no_rag_start
    
    total_latency_ms = round((retrieval_time + generation_time + no_rag_time) * 1000, 1)
    
    # Get stats
    stats = pipeline.get_stats()
    
    return jsonify({
        "query": query,
        "results": response.retrieved_chunks,
        "prompt": response.prompt,
        "answer": generated_response,
        "no_rag_answer": no_rag_response,
        "stats": {
            "num_chunks": stats["num_chunks"],
            "avg_chunk_words": round(stats["avg_chunk_words"], 1)
        },
        "latency_ms": total_latency_ms,
        "retrieval_ms": round(retrieval_time * 1000, 1),
        "generation_ms": round(generation_time * 1000, 1),
        "no_rag_generation_ms": round(no_rag_time * 1000, 1)
    })


@app.route('/api/chunking-comparison')
def api_chunking_comparison():
    # Combine all pages into one text for comparison
    full_text = "\n\n".join(load_pdf_pages())
    
    results = compare_chunking_strategies(full_text)
    return jsonify(results)


@app.route('/api/evaluate')
def api_evaluate():
    from evaluation import RetrievalEvaluator, load_evaluation_queries
    
    try:
        eval_queries = load_evaluation_queries("eval_queries.json")
    except FileNotFoundError:
        return jsonify({"error": "Evaluation queries file not found"}), 404
    
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    
    configurations = [
        {"name": "Fixed-200 + Hybrid", "strategy": "fixed", "params": {"chunk_size": 200, "overlap": 50}},
        {"name": "Sentence + Hybrid", "strategy": "sentence", "params": {}},
        {"name": "Semantic + Hybrid", "strategy": "semantic", "params": {}},
        {"name": "Recursive-300 + Hybrid", "strategy": "recursive", "params": {"chunk_size": 300}},
        {"name": "Recursive-500 + Hybrid", "strategy": "recursive", "params": {"chunk_size": 500}},
    ]
    
    results = []
    pages_text = load_pdf_pages()
    
    for config in configurations:
        # Create chunker and chunks
        chunker = get_chunker(config["strategy"], **config["params"])
        chunks = chunker.chunk_document(pages_text)
        
        # Create retriever
        retriever = RAGRetriever(use_reranking=True)
        
        # Evaluate
        report = evaluator.evaluate_configuration(
            config_name=config["name"],
            chunks=chunks,
            retriever=retriever,
            eval_queries=eval_queries,
            chunking_strategy=config["strategy"],
            retrieval_params=config["params"]
        )
        
        results.append(report.to_dict())
    
    return jsonify({
        "num_queries": len(eval_queries),
        "configurations": results
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

