from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation',
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name: str = Field(description='Name of Person')
    age:int = Field(gt=0, description='Age of person')
    location: str = Field(description='Place of person')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate a frictional hero name, age and location of {place} country. \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


# normal method without chain
# prompt = template.invoke({'place':'Russia'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

# using chain

chain = template | model | parser

result = chain.invoke({'place':'canada'})

print(result)