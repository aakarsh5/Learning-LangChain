from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    #repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Give 5 line summary on {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Everest'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result2 = model.invoke(prompt2)

print(result.content)

print(result2.content)