from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

#Langsmith API key and tracing configuration
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MY-FIRST-RAG-PROJECT"

# Get pdf file directory
pdf_dir = os.path.abspath("data/AI_Engineer_Roadmap_2026.pdf")   

# Load the PDF document
pdf = PyPDFLoader("data/AI_Engineer_Roadmap_2026.pdf")
docs = pdf.load()

# Split the text present in the PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100,add_start_index=True,)

documents = text_splitter.split_documents(docs)
print(f"Split blog post into {len(documents)} sub-documents.")

# Create embedding for the text chunks using Ollama embeddings
embeddings = OllamaEmbeddings(model = 'nomic-embed-text')

# Create a FAISS vector store from the documents and their embeddings
db = FAISS.from_documents(documents, embeddings)

print("RAG pipeline setup completed successfully!")