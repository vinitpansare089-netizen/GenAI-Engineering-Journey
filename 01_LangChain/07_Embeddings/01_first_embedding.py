
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "Krishna teaches Arjuna about duty.",
    "A person should fulfil their responsibilities.",
    "Python is a programming language."
]

# Generate embeddings
vectors = embeddings.embed_documents(sentences)
# vectors = embeddings.embed_query(sentences) for one query

# Compare first sentence with the other two
similarity_1 = cosine_similarity(
    [vectors[0]], [vectors[1]]
)[0][0]

similarity_2 = cosine_similarity(
    [vectors[0]], [vectors[2]]
)[0][0]

print("Sentence 1 vs Sentence 2:", similarity_1)
print("Sentence 1 vs Sentence 3:", similarity_2)
