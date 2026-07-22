# 🧠 Multimodal RAG Document Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** application built with **LangChain**, **Docling**, **ChromaDB**, and **Mistral AI**.

Upload documents in multiple formats and ask natural language questions. The application uses **Docling's layout-aware document parsing** to preserve document structure, tables, headings, and reading order for more accurate retrieval.

---

## ✨ Features

* 📄 Upload and query multiple document formats:

  * PDF
  * DOCX
  * PPTX
  * XLSX
  * HTML
  * Markdown
  * PNG, JPG, JPEG, BMP, TIFF
* 🧩 Layout-aware parsing using **Docling**
* 📚 Structure-aware chunking with **HybridChunker**
* 🔍 MMR (Maximal Marginal Relevance) retrieval for diverse context
* 🤖 Local sentence-transformer embeddings
* 💬 Natural language question answering with **Mistral AI**
* 📑 Source section citations with every answer
* 💾 Persistent Chroma vector database
* 🗂️ Conversation history within the session
* ⚡ Interactive Streamlit interface

---

## 🛠️ Tech Stack

* LangChain
* Docling
* ChromaDB
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* Mistral AI
* Streamlit

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG_Project
```

### 2. Create and activate a virtual environment

```bash
uv venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
MISTRAL_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
RAG_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── chroma_db/
```

---

## 📖 How It Works

1. Upload a supported document.
2. Docling extracts the document while preserving layout, headings, tables, and reading order.
3. HybridChunker creates structure-aware chunks.
4. HuggingFace embeddings convert chunks into vectors.
5. ChromaDB stores the vector embeddings.
6. User questions retrieve the most relevant chunks using MMR.
7. Mistral AI generates answers grounded only in the retrieved document context.

---

## 📌 Future Improvements

* Multi-document collections
* OCR support for scanned PDFs
* Metadata-based filtering
* Streaming responses
* Hybrid keyword + semantic search
* Citation highlighting
* Document comparison
* Persistent chat history
