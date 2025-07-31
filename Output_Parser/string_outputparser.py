from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    task= 'text-generation'
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template='Give your views on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Give 5 line summary on {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model

result = chain.invoke({'topic':'AI'})

print(result.content)