from flask import Flask, request, render_template_string
import torch
from pypdf import PdfReader
import main


DATA_FILE = "Machine_learning.pdf"


def load_index(data_file):
    reader = PdfReader(data_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    pages = [p.replace('\n', ' ') for p in pages if p.strip()
             and not p.strip().startswith("References")
             and not p.strip().startswith("External links")]

    chunks = []
    for p in pages:
        chunks.extend(main.chunk_text(p, max_words=150))

    embedder, faiss_index, embeddings = main.build_index(chunks)
    use_gpu = torch.cuda.is_available()
    gen = main.generator_pipeline(device=use_gpu)

    return chunks, embedder, faiss_index, gen


app = Flask(__name__)

# Load index at startup (simple and synchronous)
chunks, embedder, faiss_index, gen = load_index(DATA_FILE)


TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Simple RAG Demo</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; }
      textarea { width:100%; height:80px; }
      .context { background:#f6f6f6; padding:0.75rem; border-radius:6px; margin-bottom:0.5rem; }
      .score { color:#666; font-size:0.9rem }
      .answer { white-space:pre-wrap; background:#fffbe6; padding:1rem; border-radius:6px }
    </style>
  </head>
  <body>
    <h1>Simple RAG Demo</h1>
    <p><strong>Embedder:</strong> {{ embed_model }} &nbsp; <strong>Generator:</strong> {{ gen_model }}</p>
    <form method="post">
      <label for="query">Question</label>
      <textarea id="query" name="query">{{ query }}</textarea>
      <p><button type="submit">Ask</button></p>
    </form>

    {% if retrieved %}
      <h2>Retrieved contexts</h2>
      {% for idx, score, text in retrieved %}
        <div class="context">
          <div class="score">score={{ '%.4f'|format(score) }}</div>
          <div>{{ text }}</div>
        </div>
      {% endfor %}
    {% endif %}

    {% if answer is not none %}
      <h2>Answer</h2>
      <div class="answer">{{ answer }}</div>
    {% endif %}
  </body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    query = ''
    retrieved = None
    answer = None
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            retrieved = main.retrieve(query, embedder, faiss_index, chunks, top_k=3)
            context_texts = [t for _, _, t in retrieved]
            context = "\n\n".join(context_texts)

            prompt = (
                "Use the following extracted passages from a document to answer the question.\n"
                "If the answer is not contained in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")

            out = gen(prompt, max_length=200, num_return_sequences=1, truncation=True)
            answer_text = out[0].get('generated_text', '')
            if answer_text.startswith(prompt):
                answer = answer_text[len(prompt):].strip()
            else:
                answer = answer_text.strip()

    gen_model = getattr(gen, 'model_name', 'unknown')
    embed_model = getattr(embedder, 'model_name_or_path', getattr(embedder, 'model_name', 'sentence-transformers'))
    return render_template_string(TEMPLATE, query=query, retrieved=retrieved, answer=answer, embed_model=embed_model, gen_model=gen_model)


if __name__ == '__main__':
    # Run locally for development
    app.run(host='127.0.0.1', port=5000, debug=True)
