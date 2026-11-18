
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "01_LangChain/05_Document_Loaders/data/krishna_story.pdf"
)

documents = loader.load()

print("Total pages:", len(documents))

print("\nFirst page content:")
print(documents[0].page_content)

print("\nFirst page metadata:")
print(documents[0].metadata)
