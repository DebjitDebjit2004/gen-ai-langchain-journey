from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('cricket.pdf')

docs = loader.load()
print("=======================================================")
print(docs[0].page_content)
print("=======================================================")
print(docs[0].metadata)
print("=======================================================")

