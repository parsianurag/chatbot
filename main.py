import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if LANGCHAIN_API_KEY exists and set it
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
else:
    st.error("LANGCHAIN_API_KEY is not set. Please check your .env file.")

# Set Langsmith Tracking environment variables
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, llm_model, temperature, max_tokens):
    llm = Ollama(model=llm_model)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI")

# Select the OpenAI model
llm_model = st.sidebar.selectbox("Select Open Source model", ["mistral"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm_model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input.")
