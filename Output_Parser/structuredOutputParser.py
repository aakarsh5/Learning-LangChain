from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import ResponseSchema,StructuredOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

Schema = [
    ResponseSchema(name ='fact_1', description='fact 1 about the topic'),
    ResponseSchema(name ='fact_2', description='fact 2 about the topic'),
    ResponseSchema(name ='fact_3', description='fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(Schema)

template = PromptTemplate(
    template='Give 3 facts about {fact} \n{format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions' : parser.get_format_instructions()},
)
########################################################
# without chain normal method
# prompt = template.invoke({'fact':'Everest'});

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)


########################################################
#with chain

chain = template | model | parser

result = chain.invoke({'fact':'Everest'})

print(result)

# can not validate data as string or int 
