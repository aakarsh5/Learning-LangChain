from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load Document
loader = TextLoader('doc.txt')
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overloap = 50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformer/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs,embeddings)

# Create a retriever
retriever = vector_store.as_retriever()

# Manual query
query = "What is the summary of the document"
retrieved_doc = retriever._get_relevant_documents(query)

# Combine text
retrived_text = "\n".join([doc.page_content for doc in retrieved_doc])

# Define the llm 
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

# Prompt
prompt = "Based on following text answer the question {query} \n {retrived_text}"

# Invoke
result = llm.invoke(prompt)

print(result)



