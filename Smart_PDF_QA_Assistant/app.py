from dotenv import load_dotenv
from rag_pipeline import db, embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_llms import Ollama
import streamlit as st
import os

load_dotenv()  # Load environment variables from .env file

#Langsmith API key and tracing configuration
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"

# Create a Prompt Template for the question-answering task
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful AI."),
    ("human", "{question}")
])

# Initialize the Ollama LLM
llm = Ollama(model = 'llama3')
chain = prompt_template | llm

# Streamlit app
st.title("Smart PDF QA Assistant")
st.write("Ask questions about the PDF document!")

question = st.text_input("Enter your question here:")


if question:
    # Retrieve relevant documents from the FAISS vector store
    relevant_docs = db.similarity_search(question, k=3)
    
    # Generate an answer using the retrieved documents and the LLM
    answer = chain.run(question=question, context=relevant_docs)
    
    st.write("Answer:")
    st.write(answer)