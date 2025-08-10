from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
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


parser = StrOutputParser()

classifier_chain = prompt | model | parser2

result = classifier_chain.invoke({'feedback':'The product is better to use in this price range'})

print(result)