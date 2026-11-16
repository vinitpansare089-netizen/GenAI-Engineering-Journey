import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature= 1,
    max_tokens=100
    
)

response = llm.invoke("Explain Retrieval Augmented Generation in detail with an example.")

print(response.content)
print(response.response_metadata)