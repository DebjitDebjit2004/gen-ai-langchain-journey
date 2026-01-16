from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='CS',
    glob='*.pdf',
    loader_cls= PyPDFLoader
)

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    separator=' |||| '
)

result = splitter.split_documents(docs)

for i in result:
    print("=============================================")
    print(i.page_content)
    print("=============================================")