from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.storage import InMemoryStore as InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI

import pandas as pd
import faiss
import streamlit as st
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY") or "your_groq_key"
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# Initialize LLM
llm = ChatOpenAI(
    model="llama3-70b-8192",
    temperature=0
)

# Streamlit UI
st.title("Welcome to the Magic Show!")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV and preview
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of the uploaded data:")
    st.write(data.head())

    # Save uploaded file to a temp location for LangChain CSVLoader
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load CSV as documents
    loader = CSVLoader(file_path="temp.csv")
    docs = loader.load_and_split()

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim = len(embedding_model.embed_query("test"))

# Create FAISS vector store from docs (correct and simple)
    vector_store = FAISS.from_documents(docs, embedding_model)


    # Create retriever
    retriever = vector_store.as_retriever()

    # Define system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # User input
    user_query = st.text_input("Ask something about the CSV:")

    if user_query:
        response = rag_chain.invoke({"input": user_query})
        st.subheader("Answer:")
        st.write(response['answer'])
