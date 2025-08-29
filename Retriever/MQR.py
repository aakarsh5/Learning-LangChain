#multi query retriever

from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation'
)

docs = [
    Document(page_content="The Eiffel Tower is located in Paris and is one of the most famous landmarks in the world."),
    Document(page_content="Python is a popular programming language known for its simplicity and readability."),
    Document(page_content="The Great Wall of China is a historic fortification stretching thousands of miles across northern China."),
    Document(page_content="Machine learning is a subset of artificial intelligence focused on building systems that learn from data."),
    Document(page_content="The Colosseum in Rome was used for gladiatorial contests and public spectacles in ancient times."),
    Document(page_content="Basketball is a sport played by two teams of five players each, aiming to score points by shooting a ball through a hoop."),
    Document(page_content="The Leaning Tower of Pisa is famous for its unintended tilt and is located in Italy."),
    Document(page_content="Quantum computing leverages quantum mechanics to perform computations much faster than classical computers for some tasks."),
    Document(page_content="The Amazon rainforest is the largest tropical rainforest in the world and is home to diverse species."),
    Document(page_content="Soccer, also known as football, is the most popular sport in the world, played by millions globally.")
]

query = "Famous historical monuments in Europe"

embedding = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding,
)

#retriever
similarity_retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={'k':3})

#multi query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
    llm = llm
)

#result
similarity_result = similarity_retriever.invoke(query)
mqr_result = multi_query_retriever.invoke(query)

for doc in similarity_result:
    print(doc.page_content)

print("MQR Result\f")

for doc in mqr_result:
    print(doc.page_content)
