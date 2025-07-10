from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#text based embedding
# text = "Luffy is Pirate"

# vector = embedding.embed_query(text)

#document based embedding
document = [
    "What is AI",
    "What is capital of India"
]

vector = embedding.embed_documents(document)

print(str(vector))
