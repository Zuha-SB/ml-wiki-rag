from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import argparse
import sys


# Simple RAG flow:
# 1) Read PDF and extract text
# 2) Chunk text into smaller passages
# 3) Build TF-IDF index of chunks
# 4) For a query, retrieve top-k similar chunks
# 5) Construct a prompt containing retrieved context and generate an answer


def chunk_text(text, max_words=150):
    """Naive chunking: split by sentences and accumulate up to max_words."""
    chunks = []
    if not text or not text.strip():
        return chunks
    sentences = [s.strip() for s in text.replace('\n', ' ').split('. ') if s.strip()]
    current = []
    current_count = 0
    for s in sentences:
        words = s.split()
        if current_count + len(words) <= max_words:
            current.append(s)
            current_count += len(words)
        else:
            if current:
                chunks.append('. '.join(current).strip() + '.')
            # start new chunk
            current = [s]
            current_count = len(words)
    if current:
        chunks.append('. '.join(current).strip() + '.')
    return chunks


def build_index(chunks, model_name: str = "all-MiniLM-L6-v2"):
    """
    Build a FAISS index over sentence-transformers embeddings for the given chunks.
    Returns (embedder, faiss_index, embeddings_numpy)
    """
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # normalize for cosine similarity using inner product index
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return embedder, index, embeddings


def retrieve(query, embedder, index, chunks, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    # D are inner products (cosine similarities because of normalization)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append((int(idx), float(score), chunks[int(idx)]))
    return results


def generator_pipeline(device=None):
    """
    Try to load `Xenova/gpt-3.5-turbo` via HF transformers. If that fails,
    fall back to a small `gpt2` pipeline. Returns a callable that mirrors the
    HF `pipeline(..., task='text-generation')` output shape (list of dicts
    with key `generated_text`).
    """

    class HFGenerator:
        def __init__(self, tokenizer, model):
            self.tokenizer = tokenizer
            self.model = model
            # record a friendly model name for reporting
            self.model_name = getattr(tokenizer, "name_or_path", None) or getattr(model, "name_or_path", "Xenova/gpt-3.5-turbo")

        def __call__(self, prompt, max_length=200, num_return_sequences=1, truncation=True):
            # Tokenize and move inputs to model device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            gen_out = self.model.generate(**inputs, max_length=max_length, num_return_sequences=num_return_sequences)
            texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen_out]
            return [{"generated_text": t} for t in texts]

    # prefer smaller open-source 20B model first
    device_id = 0 if (device is not None and device) else -1
    try:
        pipe = pipeline("text-generation", model="openai/gpt-oss-20b", device=device_id)
        try:
            pipe.model_name = "openai/gpt-oss-20b"
        except Exception:
            pass
        return pipe
    except Exception:
        # try Xenova path as a next option
        try:
            tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-3.5-turbo")
            try:
                model = AutoModelForCausalLM.from_pretrained("Xenova/gpt-3.5-turbo")
            except Exception:
                model = AutoModel.from_pretrained("Xenova/gpt-3.5-turbo", dtype="auto")

            if torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")

            return HFGenerator(tokenizer, model)
        except Exception:
            # final fallback to local gpt2 pipeline
            pipe = pipeline("text-generation", model="gpt2", device=device_id)
            try:
                pipe.model_name = "gpt2"
            except Exception:
                pass
            return pipe


def main(args):
    data_file = args.file
    reader = PdfReader(data_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    # Basic cleaning and remove end sections
    pages = [p.replace('\n', ' ') for p in pages if p.strip()
             and not p.strip().startswith("References")
             and not p.strip().startswith("External links")]

    # Chunk each page and keep track of source
    chunks = []
    for i, p in enumerate(pages):
        for c in chunk_text(p, max_words=150):
            chunks.append(c)

    if not chunks:
        print("No text chunks extracted from the document.")
        sys.exit(1)

    embedder, faiss_index, embeddings = build_index(chunks)
    embed_model_name = getattr(embedder, "model_name_or_path", None) or getattr(embedder, "model_name", "sentence-transformers")
    print(f"Embedding model: {embed_model_name}")

    # choose device if available
    use_gpu = torch.cuda.is_available()
    gen = generator_pipeline(device=use_gpu)
    gen_model_name = getattr(gen, "model_name", None) or getattr(gen, "tokenizer", None) and getattr(gen.tokenizer, "name_or_path", None) or "unknown-generator"
    print(f"Generator model: {gen_model_name}")

    query = args.query
    if not query:
        # interactive if no query passed
        try:
            query = input("Enter your question: ")
        except KeyboardInterrupt:
            print()
            return

    top_k = args.top_k
    retrieved = retrieve(query, embedder, faiss_index, chunks, top_k=top_k)

    print("\nRetrieved contexts (top {}):\n".format(top_k))
    context_texts = []
    for idx, score, text in retrieved:
        print(f"- (score={score:.4f}) {text[:300].strip()}...\n")
        context_texts.append(text)

    # build a concise context
    context = "\n\n".join(context_texts)

    prompt = (
        "Use the following extracted passages from a document to answer the question.\n"
        "If the answer is not contained in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")

    # generate
    out = gen(prompt, max_length=200, num_return_sequences=1, truncation=True)
    answer = out[0]["generated_text"]

    # The pipeline returns the whole prompt + completion, so strip the prompt prefix
    if answer.startswith(prompt):
        completion = answer[len(prompt):].strip()
    else:
        completion = answer.strip()

    print("\nAnswer:\n")
    print(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RAG demo using TF-IDF retrieval and a generator")
    parser.add_argument("--file", "-f", default="Machine_learning.pdf", help="PDF data file")
    parser.add_argument("--query", "-q", default=None, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieved chunks")
    args = parser.parse_args()
    main(args)
