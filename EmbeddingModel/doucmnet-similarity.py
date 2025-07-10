from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = [
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "New Delhi is the capital city of India.",
    "The sun rises in the east and sets in the west.",
    "Python is a popular programming language for data science.",
    "Mount Everest is the highest mountain in the world."
]

query = "What is the tallest mountain on Earth?"

doc_embedding = embedding.embed_documents(document)
query_embedding = embedding.aembed_query(query)

print(cosine_similarity([query_embedding],doc_embedding))