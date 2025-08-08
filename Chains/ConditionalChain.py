from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template='Give the sentiment of user review positive or negative \n {feedback}',
    input_variables=['feedback']
)

parser = StrOutputParser()

classifier_chain = prompt | model | parser

result = classifier_chain.invoke({'feedback':'The product is better to use in this price range'})

print(result)