import os
import re
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load Documents
documents_path = "dataset/documents"

documents = []
ids = []
metadatas = []

for file in sorted(os.listdir(documents_path)):
    if file.endswith(".txt"):
        with open(os.path.join(documents_path, file), "r", encoding="utf-8") as f:
            documents.append(f.read())
            ids.append(file)
            metadatas.append({
                "title": file.replace(".txt", ""),
                "category": "Education",
                "source": file
            })

print("Documents Loaded Successfully!")
print("Total Documents:", len(documents))

# Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

print("Embeddings Generated Successfully!")
print("Embedding Shape:", embeddings.shape)

# Save Embeddings as NumPy File
np.save("dataset/embeddings.npy", embeddings)

print("Embeddings Saved Successfully!")

# Create ChromaDB Collection
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="education_documents"
)

print("ChromaDB Collection Created Successfully!")

# Store Documents, Embeddings and Metadata
collection.upsert(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=ids,
    metadatas=metadatas
)

print("Documents, Embeddings and Metadata Stored Successfully!")
print("Total Stored Documents:", collection.count())

# 10 User Queries
queries = [
    "How does education help students?",
    "What are the benefits of online learning?",
    "How is artificial intelligence used in education?",
    "Why is classroom learning important?",
    "How does technology improve education?",
    "What is the importance of higher education?",
    "How do teachers evaluate student performance?",
    "What is inclusive education?",
    "What are the benefits of distance education?",
    "How can students improve their study habits?"
]

# Save Queries as CSV
os.makedirs("dataset", exist_ok=True)

query_df = pd.DataFrame({"Query": queries})
query_df.to_csv("dataset/queries.csv", index=False)

print("Queries Saved Successfully!")

comparison_data = []
retrieved_data = []

# Keyword and Semantic Search
for query in queries:

    print("\nUSER QUERY:", query)

    # Semantic Search
    query_embedding = model.encode([query])

    semantic_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    print("\nSemantic Search Results:")

    for i, doc in enumerate(semantic_results["documents"][0]):
        source = semantic_results["metadatas"][0][i]["source"]
        distance = semantic_results["distances"][0][i]

        print(i + 1, source, "Distance:", distance)

        retrieved_data.append({
            "Query": query,
            "Rank": i + 1,
            "Source": source,
            "Document": doc,
            "Category": semantic_results["metadatas"][0][i]["category"],
            "Distance": distance
        })

        comparison_data.append({
            "Query": query,
            "Search_Type": "Semantic Search",
            "Rank": i + 1,
            "Source": source,
            "Document": doc,
            "Score": distance
        })

    # Keyword Search
    query_words = set(re.findall(r'\w+', query.lower()))

    keyword_results = []

    for i, doc in enumerate(documents):
        doc_words = set(re.findall(r'\w+', doc.lower()))
        score = len(query_words.intersection(doc_words))

        keyword_results.append((score, ids[i], doc))

    keyword_results.sort(key=lambda x: x[0], reverse=True)

    print("\nKeyword Search Results:")

    for rank, (score, source, doc) in enumerate(keyword_results[:5], 1):
        print(rank, source, "Score:", score)

        comparison_data.append({
            "Query": query,
            "Search_Type": "Keyword Search",
            "Rank": rank,
            "Source": source,
            "Document": doc,
            "Score": score
        })

# Save Retrieved Results
retrieved_df = pd.DataFrame(retrieved_data)
retrieved_df.to_csv("dataset/retrieved_results.csv", index=False)

# Save Keyword and Semantic Search Comparison
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("search_comparison.csv", index=False)

print("\n10 Queries Completed Successfully!")
print("Top-5 Retrieved Results Saved Successfully!")
print("Keyword and Semantic Search Comparison Saved!")
print("Embeddings, Queries and Retrieval Results Saved Successfully!")