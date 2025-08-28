from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain is a framework for building applications powered by language models.", metadata={"source": "intro", "id": 1}),
    Document(page_content="The Document class stores text and metadata.", metadata={"source": "docs", "id": 2}),
    Document(page_content="You can chunk documents before feeding them to an LLM.", metadata={"source": "docs", "id": 3}),
    Document(page_content="Metadata helps track source, author, or other useful attributes.", metadata={"source": "guide", "id": 4}),
    Document(page_content="LangChain supports integrations with many vector databases.", metadata={"source": "reference", "id": 5}),
]

embeddings = HuggingFaceEmbeddings()

#creating chroma store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name='first_collection'
)

#make retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

query = "What is langchain"
result = retriever.invoke(query)

print(result[0].page_content)
print(len(result))
