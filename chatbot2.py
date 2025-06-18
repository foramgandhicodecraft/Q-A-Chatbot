from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A chatbot using ollama"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the questions asked"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, engine):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

temperature = st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.title("Simple Q&A chatbot using Ollama")

st.write("Ask any question")
user_input = st.text_input("Type here")

engine = st.sidebar.selectbox("Select open source model",["mistral","llama3.3","gemma3"])

if user_input:
    response = generate_response(user_input, engine)
    st.write(response)