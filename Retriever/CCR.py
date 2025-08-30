# context compression retriever
# give only part which is relevant to query from docs
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from langchain_core.documents import Document

docs = [
    Document(
        page_content=(
            "The Eiffel Tower is located in Paris, France. "
            "It was completed in 1889 and designed by Gustave Eiffel. "
            "The tower attracts millions of tourists every year. "
            "Paris also has many famous landmarks like the Louvre Museum and Notre Dame Cathedral."
        ),
        metadata={"source": "travel_guide", "page": 12}
    ),
    Document(
        page_content=(
            "The Great Wall of China stretches over 13,000 miles. "
            "Construction began as early as the 7th century BC. "
            "It is often mistakenly compared to modern monuments like the Eiffel Tower, "
            "even though they serve very different purposes."
        ),
        metadata={"source": "history_textbook", "chapter": 3}
    ),
    Document(
        page_content=(
            "Python is a popular programming language. "
            "It has nothing to do with historical monuments like the Eiffel Tower, "
            "but many students learn Python when studying data science and artificial intelligence."
        ),
        metadata={"source": "tech_magazine", "issue": "July 2023"}
    ),
    Document(
        page_content=(
            "Tesla Inc., founded in 2003, is an American electric vehicle company. "
            "Although unrelated to the Eiffel Tower, Elon Musk once compared his company’s ambitious goals "
            "to humanity’s great engineering achievements, such as space travel and iconic landmarks."
        ),
        metadata={"source": "business_news", "id": 5678}
    ),
]

#  query
query = "When was the Eiffel Tower completed?"

embedding = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(docs,embedding)





