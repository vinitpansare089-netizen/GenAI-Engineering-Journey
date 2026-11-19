"""
Document Splitter in langchain
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = Document(
    page_content="""
    Krishna taught Arjuna about performing his duty.
    He explained that people should focus on their actions.

    Krishna also explained the importance of devotion.
    Faith and wisdom help people make better decisions.
    """,
    metadata={
        "source": "krishna_story.pdf",
        "page": 1
    }
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = splitter.split_documents([document])

print('Total Chunks: ',len(chunks))