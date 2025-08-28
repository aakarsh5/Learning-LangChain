#Maximun Marginal Relivance
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain is a framework for building applications powered by language models.", metadata={"source": "intro", "id": 1}),
    Document(page_content="The Document class stores text and metadata.", metadata={"source": "docs", "id": 2}),
    Document(page_content="You can chunk documents before feeding them to an LLM.", metadata={"source": "docs", "id": 3}),
    Document(page_content="Metadata helps track source, author, or other useful attributes.", metadata={"source": "guide", "id": 4}),
    Document(page_content="LangChain supports integrations with many vector databases.", metadata={"source": "reference", "id": 5}),
]

Embeddings = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=Embeddings
)

#using MMR in retriever
retriever = vectorstore.as_retriever(
    search_type ="mmr",
    search_kwargs={'k':3, "lambda_mult":1} #lambda_mult is parameter for diversity
)

query = "what is langchain"
result = retriever.invoke(query)
print(len(result))
for r in result:
    print(r.page_content)
    
