

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import json
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
load_dotenv()


# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_TYPES = [
    "pdf", "docx", "pptx", "xlsx",
    "html", "htm",
    "png", "jpg", "jpeg", "bmp", "tiff",
    "md",
]
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="🧠")

st.title("🧠 Multimodal RAG Assistant")
st.write("Upload a document and ask questions from it.")
st.caption("Supported: PDF · DOCX · PPTX · XLSX · HTML · PNG · JPG · BMP · TIFF · Markdown")


# ── Helper: extract human-readable source label from Docling metadata ─────────
def _source_label(metadata: dict, fallback: str) -> str:
    """
    Docling attaches dl_meta to each chunk with:
      origin.filename  — the original file name
      headings         — list of section headings the chunk belongs to

    This builds a clean label like:
      "report.pdf  —  Section 3: Results"
    or falls back to just the filename.
    """
    dl_meta = metadata.get("dl_meta", {})
    if isinstance(dl_meta, str):
        try:
            dl_meta = json.loads(dl_meta)
        except (json.JSONDecodeError, TypeError):
            dl_meta = {}

    origin   = dl_meta.get("origin", {}) if isinstance(dl_meta, dict) else {}
    filename = origin.get("filename") or os.path.basename(
        metadata.get("source", fallback)
    )

    headings = dl_meta.get("headings", []) if isinstance(dl_meta, dict) else []
    heading  = headings[0] if headings else None

    if filename and heading:
        return f"{filename}  —  {heading}"
    return filename or fallback



uploaded_file = st.file_uploader(
    "Upload a document",
    type=SUPPORTED_TYPES,
    help="Docling parses layout, tables, and reading order for all these formats.",
)


if uploaded_file:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
    st.info(
        f"Detected: **{suffix.lstrip('.').upper()}**"
        " — Docling will parse layout, tables, and reading order automatically."
    )

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            loader = DoclingLoader(
                file_path=file_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
            )
            chunks = loader.load()
            for chunk in chunks:
                if "dl_meta" in chunk.metadata:
                    chunk.metadata["dl_meta"] = json.dumps(chunk.metadata["dl_meta"])
            if not chunks:
                st.error("No content could be extracted from the document.")
                st.stop()
        st.write(f"📄 Extracted **{len(chunks)} chunks** from the document.")    
        with st.spinner("Building vector database…"):
            embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            vectorstore.persist()
        st.session_state["source_filename"] = uploaded_file.name
        st.success("Vector database created!")



if os.path.exists("chroma_db"):

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    # ── ADDITION 1: Conversation history ─────────────────────────────────────
    # Initialise chat history in session state so it persists across reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display all previous Q&A turns before the input box
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            # Re-render citations for past turns
            with st.expander("📄 Sources"):
                for src in turn["sources"]:
                    st.caption(f"**{src['label']}**  —  {src['preview']}")
    # ─────────────────────────────────────────────────────────────────────────

    query = st.chat_input("Ask a question about the document")

if query:

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    # Generate answer
    response = llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.write(response.content)

    # Source citations
    sources = []
    seen    = set()
    fallback_name = st.session_state.get("source_filename", "document")

    for doc in docs:
        label = _source_label(doc.metadata, fallback_name)
        if label not in seen:
            seen.add(label)
            sources.append({
                "label":   label,
                "preview": doc.page_content[:120].replace("\n", " ") + "…",
            })

    with st.expander("📄 Sources"):
        for src in sources:
            st.caption(f"**{src['label']}**  —  {src['preview']}")


    # Save conversation
    st.session_state.chat_history.append(
        {
            "question": query,
            "answer": response.content,
            "sources": sources
        }
    )
        # ─────────────────────────────────────────────────────────────────────