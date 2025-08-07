from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation',
)

Prompt = PromptTemplate(
    template='Give 3 interesting fact about {topic}',
    input_variables=['topic'],
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = Prompt | model | parser

result = chain.invoke({'Japan'})

print(result)