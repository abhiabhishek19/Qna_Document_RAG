

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            vectorstore.persist()

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
                    st.caption(f"Page {src['page']} — {src['preview']}")
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
    seen = set()

    for doc in docs:

        page = doc.metadata.get("page", "Unknown")

        if page not in seen:

            seen.add(page)

            sources.append(
                {
                    "page": page + 1 if isinstance(page, int) else page,
                    "preview": doc.page_content[:120].replace("\n", " ") + "..."
                }
            )

    with st.expander("📄 Sources"):
        for src in sources:
            st.caption(
                f"Page {src['page']} — {src['preview']}"
            )

    # Save conversation
    st.session_state.chat_history.append(
        {
            "question": query,
            "answer": response.content,
            "sources": sources
        }
    )
        # ─────────────────────────────────────────────────────────────────────