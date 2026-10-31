
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Creating my prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the questions asked."),
    ("user", "Question: {question}")
])

# Streamlit framework
st.title("LangChain Demo with Gemma 2B")
input_text = st.text_input("What question do you have in mind?")

# Ollama model
llm = ChatOllama(model="gemma2:2b")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
