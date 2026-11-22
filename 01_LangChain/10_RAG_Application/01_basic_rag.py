"""
Loading the components for RAG
"""

from dotenv import load_dotenv

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

load_dotenv()


#Embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

###Load FAISS index
vector_store = FAISS.load_local(
    "faiss_krishna_index",
    embeddings,
    allow_dangerous_deserialization=True
)

##Retriever 
retriever = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k': 3}
)

###Load groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature = 0,
    max_tokens= 1024
)