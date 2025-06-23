import os
import pandas as pd
import faiss
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer
import subprocess

INDEX_PATH = "vector.index"
TEXTS_PATH = "texts.json"
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Load or create FAISS index
def load_or_create_index(csv_path):
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        print("🔁 Loading saved FAISS index and text data...")
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, 'r') as f:
            texts = json.load(f)
        embedder = SentenceTransformer(EMBED_MODEL)
        return texts, index, embedder

    print("📦 Creating new FAISS index from CSV...")
    df = pd.read_csv(csv_path)

    # Combine only the relevant fields for context
    def row_to_text(row):
        notes = row['notes'].strip("[]").replace("'", "")
        return f"Title: {row['title']}\nDesigner: {row['designer']}\nNotes: {notes}\nDescription: {row['description']}\nRating: {row['rating']}"

    texts = df.apply(row_to_text, axis=1).tolist()

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and texts
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, 'w') as f:
        json.dump(texts, f)

    return texts, index, embedder


# Retrieve top-k similar rows
def retrieve_context(query, embedder, index, texts, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Use Ollama to get response from local LLM
def query_ollama(context, query, model='llama3'):
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def main():
    csv_path = "perfumes_table.csv"  # Replace with your CSV
    texts, index, embedder = load_or_create_index(csv_path)

    print("RAG system ready. Type your query:")
    while True:
        try:
            query = input(">>> ")
            if query.lower() in ['exit', 'quit']:
                break
            context_snippets = retrieve_context(query, embedder, index, texts)
            combined_context = "\n---\n".join(context_snippets)
            response = query_ollama(combined_context, query)
            print("\n📚 Answer:\n", response, "\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
