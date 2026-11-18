from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    '01_LangChain/05_Document_Loaders/data/krishna_stories.csv'
)

document = loader.load()

print(len(document))