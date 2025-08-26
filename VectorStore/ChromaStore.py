from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

#Creating list of document object 
docs = [
    Document(page_content="LangChain is a framework for building applications powered by language models.", metadata={"source": "intro", "id": 1}),
    Document(page_content="The Document class stores text and metadata.", metadata={"source": "docs", "id": 2}),
    Document(page_content="You can chunk documents before feeding them to an LLM.", metadata={"source": "docs", "id": 3}),
    Document(page_content="Metadata helps track source, author, or other useful attributes.", metadata={"source": "guide", "id": 4}),
    Document(page_content="LangChain supports integrations with many vector databases.", metadata={"source": "reference", "id": 5}),
]

vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(), #model which will convert document to embeddings
    persist_directory='chroma_db', #location of storage
    collection_name='sample' #collection name is sample
)

#to add document we can add doucment 
vector_store.add_documents(docs)

#view document
vector_store.get()