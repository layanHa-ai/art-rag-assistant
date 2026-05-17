# Art RAG Assistant 🎨

A simple Retrieval-Augmented Generation (RAG) application for art-related documents.

The app allows users to upload local files and ask questions using semantic search and AI-generated responses.

## Features

- Upload .txt, .md, and .pdf documents
- Semantic search using FAISS
- Context retrieval with embeddings
- AI-generated answers using OpenAI API
- Source-referenced responses
- Streamlit user interface

## Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- OpenAI API

## How It Works

1. Upload art-related documents
2. Documents are split into chunks
3. Text chunks are converted into embeddings
4. FAISS performs similarity search
5. Relevant chunks are sent to the LLM
6. The model generates a context-aware answer

## Installation

bash git clone https://github.com/layanHa-ai/art-rag-assistant.git cd art-rag-assistant pip install -r requirements.txt 

## Run the App

bash streamlit run app.py 

## Project Structure

## Project Structure


art-rag-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── documents/
└── .gitignore


## Future Improvements

- Multi-document retrieval
- Chat history memory
- Better UI design
- Support for more file types

## GitHub

Art RAG Assistant Repository
