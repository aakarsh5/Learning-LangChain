from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header('Prompt Interface')

user_input = st.text_input("Enter Your Prompt")

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceTB/SmolLM3-3B",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
    


