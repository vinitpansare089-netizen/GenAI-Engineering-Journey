from langchain_community.document_loaders import TextLoader


##Loaded the Text file
loader = TextLoader(
    '01_LangChain/05_Document_Loaders/data/krishna_story.txt',
    encoding= "utf-8"
)

###Extract the document
documents = loader.load()

print('Documents: ',len(documents))
print(documents[0].page_content)
print(documents[0].metadata)