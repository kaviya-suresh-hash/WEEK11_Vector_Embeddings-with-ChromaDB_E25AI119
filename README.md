Week 11 – Vector Embeddings with ChromaDB

Project Description

This project demonstrates how to generate vector embeddings from text documents and perform semantic search using ChromaDB.

Technologies Used

- Python
- NumPy
- Pandas
- Sentence Transformers
- ChromaDB
- Jupyter Notebook

Embedding Model

"all-MiniLM-L6-v2"

Project Structure

Vector_Embeddings/
│
├── dataset/
│   ├── documents/
│   ├── embeddings.npy
│   ├── queries.csv
│   └── retrieved_results.csv
│
├── notebook/
│   └── Week11_Vector_Embeddings.ipynb
│
├── chroma_db/
├── vector_store.py
├── search_comparison.csv
├── retrieval_results.csv
├── requirements.txt
└── README.md

Features

- Loads 10 text documents.
- Generates 384-dimensional vector embeddings.
- Stores embeddings using ChromaDB.
- Performs semantic search.
- Performs keyword search.
- Retrieves the Top-5 matching documents.
- Compares semantic search and keyword search results.

How to Run

1. Install the required packages:
   "pip install -r requirements.txt"
2. Run the Python file:
   "python vector_store.py"
3. Open the Jupyter Notebook to view the implementation and results.

Output

- Vector embeddings saved in "embeddings.npy".
- Search queries saved in "queries.csv".
- Retrieved results saved in "retrieved_results.csv".
- Search comparison saved in "search_comparison.csv".

Conclusion

This project successfully demonstrates vector embedding generation, storage using ChromaDB, and semantic document retrieval.