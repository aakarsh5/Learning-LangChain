from langchain_community.retrievers import WikipediaRetriever

#initialize the retriever
retriever = WikipediaRetriever(top_k_results=2,lang='en')

query = "Give Importance of Diwali in hindu religion"

docs = retriever.invoke(query)
print(len(docs))