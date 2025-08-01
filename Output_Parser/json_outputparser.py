from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 names of super heroes\n {format_ins}',
    input_variables=[],
    partial_variables={'format_ins':parser.get_format_instructions()}
)

#                  normal way
# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

#                using chain

chain = template | model | parser

result = chain.invoke({})

print(result)
