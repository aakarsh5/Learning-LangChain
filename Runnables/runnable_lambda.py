from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Give a joke on {topic}',
    input_variables=['topic']
)

def word_count(text):
    return len(text.split())

chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

# direct function
# parallel_chain = RunnableParallel({
#     'joke':RunnablePassthrough(),
#     'word_count':RunnableLambda(lambda x: len(x.split()))
# })

final_chain = RunnableSequence(chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n Word Count {}""".format(result['joke'], result['word_count'])

print(final_result)
