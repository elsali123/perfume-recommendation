import json
from sentence_transformers import SentenceTransformer
import numpy as np
import ast

with open('data.json', 'r') as f:
    data = json.load(f)
    
model = SentenceTransformer('all-mpnet-base-v2')
texts = []
metadata = []
i = 1
total = len(data)
for name, info in data.items():
    print(f"{i} out of {total}")
    description = info.get("description", "")
    notes = ast.literal_eval(info.get("notes", "[]"))
    designer = info.get("designer", "")
    #reviews = ast.literal_eval(info.get("reviews", "[]"))
    reviews = ast.literal_eval(info["reviews"][0]) if info.get("reviews") and isinstance(info["reviews"], list) and len(info["reviews"]) > 0 else []

    all_text = f"{name}. Description: {description}. Notes: {', '.join(notes)}. Designer: {designer}. Reviews: {', '.join(reviews)}."
    
    texts.append(all_text)
    metadata.append({"name": name, **info})
    i += 1

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True
).astype('float32')

print("embedding finished")
dimension = embeddings.shape[1]
print("dimension finished")
index = faiss.IndexFlatL2(dimension)
print("index finished")
index.add(embeddings)
print("index added")

# Save index and metadata
faiss.write_index(index, 'perfume_index.faiss')
print("faiss writte")
with open('perfume_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)