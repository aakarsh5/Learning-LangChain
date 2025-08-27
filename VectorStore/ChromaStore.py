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

# #to add document we can add doucment 
vector_store.add_documents(docs)

# #view document
print(vector_store.get())

#similarity search
# result = vector_store.similarity_search(
#     query='Define Langchain?',
#     k = 2
# )
# print(result)

#similarity search with score
# result = vector_store.similarity_search_with_score(
#     query='Define Document',
#     k = 2
# )
# print(result)

#search by metadata
# result = vector_store.similarity_search_with_score(
#     query='',
#     filter={"source": "docs"}
# )
# print(result)

#update document
# update_doc1 = Document(page_content="LangChain is an open-source framework that helps developers build applications powered by large language models (LLMs) by connecting them with data sources, APIs, and tools in a structured way.", metadata={"source": "intro", "id": 1})

# vector_store.update_document(document_id='4d32fdf6-1a1e-40e8-aaca-4a3e6235a424',document=update_doc1)

#delete document
#vector_store.delete(ids=['    '])
