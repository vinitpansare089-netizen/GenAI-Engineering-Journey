from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    '01_LangChain/05_Document_Loaders/data/krishna_stories.csv'
)

document = loader.load()
###////   loader.lazy_load()	Loads documents incrementally

print(len(document))   #?/////	Counts Document objects
print(document[0].page_content)  #//// Accesses extracted content
print(document[0].metadata)