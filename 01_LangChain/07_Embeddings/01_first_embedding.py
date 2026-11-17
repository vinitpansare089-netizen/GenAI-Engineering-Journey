from langchain_huggingface import HuggingFaceEmbeddings

##Embedding model 
embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)


###Text
text = 'Krishna is the god'

## Converted text into vectors
vector = embedding.embed_query(text)

print('Vector: ', vector)
print('dimensions: ', len(vector))
print('First 5 Values: ', vector[:5])
