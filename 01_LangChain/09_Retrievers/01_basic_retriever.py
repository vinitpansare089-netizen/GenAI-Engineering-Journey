from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

enbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "faiss_krishna_index",
    enbeddings,
    allow_dangerous_deserialization=True
)

print("Stored metadata:")

for doc in vector_store.similarity_search(
    "Krishna",
    k=5
):
    print(doc.metadata)

retriever = vector_store.as_retriever( #/// This converts your vector store into a retriever.
    search_type = 'mmr',
    search_kwargs = {
        'k' : 2,
        'fetch_k' : 5,
        'lambda_mult' : 0,
        "filter": {"deity": "Shiva"}
    }
)

query = "How can I perform my duty without worrying about results?"

results = retriever.invoke(query)

for i, doc in enumerate(results,start = 1):
    print(f'\nResult {i}')
    print('content: ', doc.page_content)
    print('metadata: ', doc.metadata)