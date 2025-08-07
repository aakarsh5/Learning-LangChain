from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'Describe in sort about {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Give 3 point summary on {text}',
    input_variables=['text'],
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Messi'})

print(result)

chain.get_graph().print_ascii()