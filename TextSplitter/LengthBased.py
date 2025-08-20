from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('../DocumentLoader/langchain.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)
# every chunk is a document object
result = splitter.split_documents(docs)

print(result[0].page_content)
print(result[0].metadata)
