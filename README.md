# Stat IQ - PDF Q&A System

## Overview
Stat IQ is a simple Retrieval-Augmented Generation (RAG) system that enables users to upload a PDF document and ask questions related to its content. The system extracts text from the uploaded PDF and leverages the Groq API to generate responses based on the document's content.

## Features
- Upload a PDF file for text extraction.
- Ask questions related to the document.
- AI-powered responses using a conversational history.
- Handles API errors and rate limits efficiently.

## How It Works
1. User uploads a PDF file.
2. The system extracts text from the document.
3. User asks a question related to the PDF content.
4. The Groq API processes the query with the extracted text as context and returns a response.
5. The response is displayed to the user.

## Future Enhancements
This repository will be updated with more advanced RAG techniques, including:
- Chunk-based retrieval for better context handling.
- Vector embeddings for semantic search.
- Hybrid search using dense and sparse retrieval methods.
- Multi-document querying with ranking algorithms.

Stay tuned for more updates on advanced RAG implementations!

