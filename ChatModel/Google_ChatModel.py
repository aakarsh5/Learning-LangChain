from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'models/gemini-1.5-pro')

result = model.invoke("Who is the prime minister of Nepal?")

print(result.content)