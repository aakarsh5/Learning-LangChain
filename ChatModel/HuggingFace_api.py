from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceTB/SmolLM3-3B",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)


result = model.invoke("Who is the richest person in Nepal?")

print(result.content)