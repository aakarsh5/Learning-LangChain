from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class student(BaseModel):
    name: str = 'Aakarsh'
    age: int = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default = 8, description = "cgpa should range from 0 to 10")

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceTB/SmolLM3-3B",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

structured_model = model.with_structured_output(student)

result = structured_model.invoke("""Aakarsh is a college student currently pursuing his undergraduate degree. He is 20 years old and maintains a solid academic record with a CGPA of 8.0, which falls within the expected range of 0 to 10. You can reach him via email at aakarsh@example.com. Aakarsh is known for his dedication to his studies and consistent performance across semesters.""")

print(result)

