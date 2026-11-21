from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

# 1. Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load existing FAISS index
vector_store = FAISS.load_local(
    "faiss_krishna_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Get our existing documents
documents = vector_store.similarity_search(
    "Krishna",
    k=5
)

# 4. Create vector retriever
vector_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 5. Create keyword retriever
bm25_retriever = BM25Retriever.from_documents(
    documents
)
bm25_retriever.k = 3

# 6. Combine both retrievers
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever
    ],
    weights=[0.5, 0.5]
)

# 7. Search
query = "Nishkama Karma and duty"

results = hybrid_retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)