from dotenv import load_dotenv
from rag_pipeline import db, embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st
import os

load_dotenv()  # Load environment variables from .env file

#Langsmith API key and tracing configuration
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MY-FIRST-RAG-PROJECT"

# Create a Prompt Template for the question-answering task
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant. Answer ONLY from the provided context. "
     "If answer is not in context, say 'I don't know'."),
     
    ("human", 
     "Context:\n{context}\n\nQuestion:\n{question}")
])

# Initialize the Ollama LLM
llm = OllamaLLM(model = 'llama3')
chain = prompt_template | llm

# Streamlit app
st.title("Smart PDF QA Assistant")
st.write("Ask questions about the PDF document!")

question = st.text_input("Enter your question here:")


if question:
    # Retrieve relevant documents from the FAISS vector store
    results = db.similarity_search_with_score(question, k=3)
    relevant_docs = [doc for doc, score in results]

    # Convert documents to text
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate an answer using the retrieved documents and the LLM
    answer = chain.invoke({
    "question": question,
    "context": context
})
    
    st.write("Answer:")
    # Display the retrieved documents and the generated answer
    # for doc,score in results:
    #     st.write("Score:", score)
    #     st.write(doc.page_content[:300])
    #     st.write("---")
    st.write(answer)