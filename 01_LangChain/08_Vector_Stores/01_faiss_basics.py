"""
Create a few Krishna-related Documents.
Split them into chunks.
Generate embeddings using MiniLM.
Store those embeddings in FAISS.
Search using a natural-language question.
Retrieve the most relevant Documents
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

###  Prepare the Document

documents = [
    Document(
        page_content="Krishna teaches Arjuna to perform his duty without attachment to results.",
        metadata={"source": "krishna_story_1.txt"}
    ),
    Document(
        page_content="Krishna explains that devotion and faith help a person find inner peace.",
        metadata={"source": "krishna_story_2.txt"}
    ),
    Document(
        page_content="Arjuna learns that wisdom and discipline are important when making difficult decisions.",
        metadata={"source": "krishna_story_3.txt"}
    ),
    Document(
        page_content="Krishna explains that anger can cloud judgement and destroy wisdom.",
        metadata={"source": "krishna_story_4.txt"}
    ),
    Document(
        page_content="A person should remain calm during success and failure.",
        metadata={"source": "krishna_story_5.txt"}
    )
]

## Splitting the Document 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

### Chunking
chunks = splitter.split_documents(documents)



