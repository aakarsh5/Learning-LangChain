from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import pydantic,PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

class Feedback(BaseModel):

    sentiment:Literal['Positive', 'Negative'] = Field(description='Give sentiment of feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt = PromptTemplate(
    template='Give the sentiment of user review positive or negative \n {feedback} \n{instruction_format}',
    input_variables=['feedback'],
    partial_variables={'instruction_format':parser2.get_format_instructions()},
)

prompt2 = PromptTemplate(
    template='Give response for positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Give response for negative feedback \n {feedback}',
    input_variables=['feedback']
)

parser = StrOutputParser()

classifier_chain = prompt | model | parser2

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser ),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser ),
    RunnableLambda(lambda x:"Couldn't analyse the sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback':'The product seems nice '}))