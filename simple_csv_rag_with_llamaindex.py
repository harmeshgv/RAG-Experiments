from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PagedCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
import faiss
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Define embedding and LLM models
EMBED_DIMENSION = 512
Settings.llm = HuggingFaceLLM(model_name="mistralai/Mistral-7B-v0.1")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load CSV file
file_path = '../data/customers-100.csv'  # Change this to your actual path
data = pd.read_csv(file_path)

# Preview the CSV file
data.head()

# Create FAISS Vector Store
faiss_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Load and Process CSV Data as Document
csv_reader = PagedCSVReader()
reader = SimpleDirectoryReader(
    input_files=[file_path],
    file_extractor={".csv": csv_reader}
)
docs = reader.load_data()

# Ingestion Pipeline
pipeline = IngestionPipeline(
    vector_store=vector_store,
    documents=docs
)
nodes = pipeline.run()

# Create Query Engine
vector_store_index = VectorStoreIndex(nodes)
query_engine = vector_store_index.as_query_engine(similarity_top_k=2)

# Query the RAG bot
response = query_engine.query("Which company does Sheryl Baxter work for?")
print(response.response)
