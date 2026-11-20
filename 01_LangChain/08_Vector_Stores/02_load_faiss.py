from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "faiss_krishna_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")
print("Total vectors:", vector_store.index.ntotal)

query = "What does Krishna teach about duty?"

results = vector_store.similarity_search(query, k=2)

for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Content:", doc.page_content)
        print("Metadata:", doc.metadata)