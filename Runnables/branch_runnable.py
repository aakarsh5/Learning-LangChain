from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template='Give a report on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt, model, parser)

prompt2 = PromptTemplate(
    template='Give Summary on {text}',
    input_variables=['text']
)

condition_chain = RunnableBranch(
    (lambda x:len(x.split())> 250, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, condition_chain)

print(final_chain.invoke({'topic':'Russia Vs Ukraine'}))
