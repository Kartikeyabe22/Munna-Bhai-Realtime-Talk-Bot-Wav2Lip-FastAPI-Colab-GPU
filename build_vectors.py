import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

with open("munna_dataset.json") as f:
    data = json.load(f)

texts = [f"{d['input']} {d['output']}" for d in data]
vectors = emb.embed_documents(texts)

np.save("dataset_vectors.npy", vectors)

print("Done")