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

retriever = vector_store.as_retriever(
    search_kwargs = {'k' : 2}
)

query = "How can I perform my duty without worrying about results?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f'\nResult {i}')
    print('content: ', doc.page_content)
    print('metadata: ', doc.metadata)