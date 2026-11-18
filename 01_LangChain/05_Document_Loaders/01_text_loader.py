from langchain_community.document_loaders import TextLoader


##Loaded the Text file
loader = TextLoader(
    '01_LangChain\05_Document_Loaders\01_text_loader.py'
)

###Extract the document
document = loader.load()

print(len(document))