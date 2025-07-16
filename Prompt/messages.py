#there are 3 types of messages 1. user message 2. system message 3.AI message
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content="You are A tutor"),
    HumanMessage(content="Tell me about Nepal")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
