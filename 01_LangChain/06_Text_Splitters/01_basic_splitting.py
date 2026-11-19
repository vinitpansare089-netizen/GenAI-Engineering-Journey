"""
Text Splitters in langchain
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Krishna taught Arjuna the importance of performing his duty.
A person should focus on their actions rather than worrying
about the results.

Krishna also explained the importance of devotion and faith.
Wisdom helps a person make better decisions in difficult situations.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = splitter.split_text(text)

print('Total Chunks: ', len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f'\n--- chunks {i} ---')
    print(chunk)
    print('characters: ',len(chunk))