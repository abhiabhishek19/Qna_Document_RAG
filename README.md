# RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) pipeline.
Upload any PDF and ask natural language questions from it.

## Features
- Upload any PDF via drag-and-drop
- MMR retrieval for diverse, non-redundant context
- Streaming responses (token by token)
- Source page citation with every answer
- Conversation history within session

## Tech Stack
LangChain · ChromaDB · HuggingFace Embeddings · Mistral AI · Streamlit

## Setup
1. Clone the repo
2. Run `uv venv` then activate
3. Run `uv add -r requirements.txt`
4. Create `.env` with  `MISTRAL_API_KEY`
5. Run `streamlit run app.py`