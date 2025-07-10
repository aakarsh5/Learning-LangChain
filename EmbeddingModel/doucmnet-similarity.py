from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(document[index])
print(score)
