from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'Give a song on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Give a joke on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'question': RunnablePassthrough(),
    'poem': RunnableSequence(prompt1, model, parser),
})

result = chain.invoke({'topic':'AI'})

print(result['question'])
print(result['poem'])