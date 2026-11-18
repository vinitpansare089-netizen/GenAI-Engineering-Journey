from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


##Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

###Document Preparation
stories = [
    "Krishna teaches Arjuna to perform his duty without attachment to results.",
    "Krishna explains the importance of devotion and faith.",
    "Python is a popular programming language."
]

##User query
text = ['How can I work without worrying about the outcome?']

###Convert stories into embeddings
story_vector = embeddings.embed_documents(stories)

###Convert query into embeddings
query_vector = embeddings.embed_query(text)

### Caalculate Cosine Similarity
scores = cosine_similarity(
    [query_vector], story_vector
)[0]

###Best Suited index find
best_index = scores.argmax()

print('user query: ', query_vector)
print('Most Reveleant Story\n')
print(stories[best_index])
print('Similarity Scores: ',scores[best_index])