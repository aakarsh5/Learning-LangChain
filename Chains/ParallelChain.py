from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation',
)

llm2 = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    task='text-generation',
)

model = ChatHuggingFace(llm = llm)

model2 = ChatHuggingFace(llm = llm2)

prompt = PromptTemplate(
    template='Give notes on {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Give answer question on {topic}',
    input_variables=['topic'],
)

prompt3 = PromptTemplate(
    template= ' Merge both notes and quiz in one document \n notes->{notes}, quiz->{quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

topic = """
LangChain** is an open-source framework designed to help developers build powerful applications using large language models (LLMs). It provides modular components and abstractions to handle tasks like prompt management, memory, chaining LLM calls, agent orchestration, and integration with external data sources (e.g., APIs, databases, file systems). LangChain supports both Python and JavaScript and is widely used for building advanced applications such as chatbots, Retrieval-Augmented Generation (RAG) systems, and autonomous agents. It integrates with various LLM providers (like OpenAI, Anthropic, Hugging Face, and local models via Ollama) and vector databases (e.g., Pinecone, FAISS, Chroma). By simplifying the development workflow and providing reusable building blocks, LangChain accelerates the creation of LLM-powered tools and applications.
"""

parallel_chain = RunnableParallel({
    'notes': prompt | model | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'topic': topic})

print(result)

chain.get_graph().print_ascii()
