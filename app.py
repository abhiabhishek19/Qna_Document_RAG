# =============================================================================
# OBSERVABILITY — add the following keys to your .env file for LangSmith tracing:
#
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=<your-langsmith-api-key>
#   LANGCHAIN_PROJECT=<your-project-name>          # e.g. "CRAG-Multimodal-RAG"
#   TAVILY_API_KEY=<your-tavily-api-key>            # required for web_search node
#   MISTRAL_API_KEY=<your-mistral-api-key>
#
# LangSmith dashboard: https://smith.langchain.com/
# =============================================================================

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil
import json

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── LangGraph CRAG graph & retriever injector ─────────────────────────────────
from logger_config import logger
import nodes                        # exposes set_retriever()
from graph import app as crag_app   # compiled LangGraph workflow
import session_manager              # isolated multi-session management

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# Silence the Docling plugin self-registration warning
os.environ["DOCLING_ALLOW_EXTERNAL_PLUGINS"] = "true"
load_dotenv()


# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_TYPES = [
    "pdf", "docx", "pptx", "xlsx",
    "html", "htm",
    "png", "jpg", "jpeg", "bmp", "tiff",
    "md",
]
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Multimodal CRAG Assistant", page_icon="🧠", layout="wide")


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


# ── Cached embedding model ────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)


# ── Multi-Session State Initialization ────────────────────────────────────────
active_session_id = session_manager.get_active_session_id()
current_session = session_manager.get_session(active_session_id)

if not current_session:
    current_session = session_manager.create_session(title="Chat 1")
    active_session_id = current_session["id"]
    session_manager.set_active_session_id(active_session_id)


# ── Sidebar: ChatGPT-like Session Navigation ──────────────────────────────────
with st.sidebar:
    st.title("🧠 Chat Sessions")
    
    # ➕ New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_sess = session_manager.create_session(title="New Chat")
        session_manager.set_active_session_id(new_sess["id"])
        st.rerun()

    st.divider()
    st.subheader("Recent Chats")
    
    sessions_list = session_manager.list_sessions()
    for sess in sessions_list:
        sid = sess["id"]
        is_active = (sid == active_session_id)
        
        col_btn, col_del = st.columns([0.82, 0.18])
        with col_btn:
            btn_label = f"💬 {sess['title']}" if not is_active else f"👉 **{sess['title']}**"
            if st.button(btn_label, key=f"sess_{sid}", use_container_width=True):
                session_manager.set_active_session_id(sid)
                st.rerun()
                
        with col_del:
            if st.button("🗑️", key=f"del_{sid}", help=f"Delete '{sess['title']}'"):
                new_act = session_manager.delete_session(sid)
                session_manager.set_active_session_id(new_act)
                st.rerun()

    st.divider()

    # Active Session Details & Management
    st.subheader("Active Session")
    st.caption(f"**Session ID:** `{active_session_id}`")
    
    # Rename chat
    new_title = st.text_input("Rename Chat", value=current_session.get("title", "New Chat"), key=f"title_{active_session_id}")
    if new_title and new_title != current_session.get("title"):
        current_session["title"] = new_title
        session_manager.update_session(current_session)
        st.rerun()

    # Active Document Status in this Session
    if current_session.get("db_ready") and current_session.get("filename"):
        st.success(f"📄 **{current_session['filename']}**\n\n({current_session.get('chunk_count', 0)} chunks indexed)")
        if st.button("🔄 Replace / Remove Document", key=f"clear_doc_{active_session_id}", use_container_width=True):
            session_manager.clear_session_db(active_session_id)
            st.rerun()
    else:
        st.info("No document attached to this chat session yet.")

    st.divider()

    # Execution Logs Viewer
    st.subheader("⚙️ System Status & Logs")
    if os.path.exists("rag_assistant.log"):
        with st.expander("📋 View Execution Logs", expanded=False):
            try:
                with open("rag_assistant.log", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    st.code("".join(lines[-35:]), language="text")
            except Exception as e:
                st.caption(f"Error reading log: {e}")
        with open("rag_assistant.log", "rb") as f:
            st.download_button(
                label="📥 Download Full Log",
                data=f,
                file_name="rag_assistant.log",
                mime="text/plain",
                use_container_width=True
            )


# ── Main Content Area ─────────────────────────────────────────────────────────
st.header(f"💬 {current_session.get('title', 'Chat')}")

session_db_dir = session_manager.get_session_db_dir(active_session_id)
db_ready = current_session.get("db_ready", False) and os.path.exists(os.path.join(session_db_dir, "chroma.sqlite3"))

# If no document is uploaded in THIS session, display the file uploader
if not db_ready:
    st.info("💡 **Upload a document for this chat session.** Each chat maintains its own isolated document context and conversation history.")
    
    uploaded_file = st.file_uploader(
        f"Upload a document for '{current_session.get('title')}'",
        type=SUPPORTED_TYPES,
        key=f"uploader_{active_session_id}",
        help="Docling parses layout, tables, and reading order for all these formats.",
    )

    if uploaded_file:
        suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")

        if st.button("Index Document for this Chat", type="primary", key=f"ingest_{active_session_id}"):
            with st.spinner("Processing document layout & chunks…"):
                loader = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=256),
                )
                chunks = loader.load()
                for chunk in chunks:
                    if "dl_meta" in chunk.metadata:
                        chunk.metadata["dl_meta"] = json.dumps(chunk.metadata["dl_meta"])
                if not chunks:
                    st.error("No content could be extracted from the document.")
                    st.stop()

            st.write(f"📄 Extracted **{len(chunks)} chunks** from the document.")

            with st.spinner("Building isolated vector store for this chat…"):
                # Clean any previous DB for this session
                session_manager.clear_session_db(active_session_id)
                session_db_dir = session_manager.get_session_db_dir(active_session_id)

                Chroma.from_documents(
                    documents=chunks,
                    embedding=get_embeddings(),
                    persist_directory=session_db_dir,
                )

            # Update session metadata
            current_session["filename"] = uploaded_file.name
            current_session["chunk_count"] = len(chunks)
            current_session["db_ready"] = True
            
            # If default title, auto-set to filename
            if current_session.get("title") in ["New Chat", "Chat 1", "Chat 2"]:
                current_session["title"] = uploaded_file.name.rsplit(".", 1)[0][:25]

            session_manager.update_session(current_session)
            st.success("✅ Document indexed! You can now ask questions below.")
            st.rerun()

    st.stop()


# ── Document Ready: Wire Session-Isolated Retriever ───────────────────────────
vectorstore = Chroma(
    persist_directory=session_db_dir,
    embedding_function=get_embeddings(),
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

# Wire this specific session's retriever into the CRAG graph
nodes.set_retriever(retriever)

# Active document banner
col_doc, col_action = st.columns([0.8, 0.2])
with col_doc:
    st.caption(f"📚 Context: **{current_session.get('filename')}** ({current_session.get('chunk_count', 0)} chunks) | Session: `{active_session_id}`")
with col_action:
    if st.button("Upload New / Replace", key=f"replace_{active_session_id}", use_container_width=True):
        session_manager.clear_session_db(active_session_id)
        st.rerun()

st.divider()

# ── Render Active Session Chat History ────────────────────────────────────────
chat_history = current_session.get("chat_history", [])

for turn in chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        datasource_tag = turn.get("datasource", "vectorstore")
        badge = "🌐 Web (Tavily)" if datasource_tag == "web" else "📚 Document Vectorstore"
        st.caption(f"Source: **{badge}**")
        with st.expander("📄 Retrieved Context Sources"):
            for src in turn.get("sources", []):
                st.caption(f"**{src['label']}** — {src['preview']}")


# ── Chat Input ────────────────────────────────────────────────────────────────
query = st.chat_input(f"Ask a question about {current_session.get('filename')}…")

if query:
    # Auto-rename chat if it's the first question and still has a generic name
    if len(chat_history) == 0 and current_session.get("title") in ["New Chat", "Chat 1", "Chat 2"]:
        current_session["title"] = session_manager.auto_title(query)
        session_manager.update_session(current_session)

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        step_labels = {
            "query_transform": "🔄 Understanding conversation context…",
            "retrieve": f"📚 Searching '{current_session.get('filename')}' vector store…",
            "grade_documents": "⚖️ Grading document relevance…",
            "web_search": "🌐 Searching the web for external info…",
            "generate": "✍️ Drafting response…",
            "check_hallucination_and_relevance": "🔍 Validating facts and grounding…",
            "increment_retries": "🔄 Refining answer…",
        }
        final_state = {
            "question": query,
            "chat_history": chat_history,
            "retries": 0,
            "web_search_attempted": False,
            "documents": [],
            "generation": "",
            "datasource": "vectorstore",
        }
        with st.status("🤔 Processing question...", expanded=True) as status:
            try:
                logger.info(
                    "=== [Session %s] New Query: '%s' (Turn %d) ===",
                    active_session_id, query, len(chat_history) + 1
                )
                for output in crag_app.stream(final_state, stream_mode="updates"):
                    for node_name, state_update in output.items():
                        label = step_labels.get(node_name, f"Completed {node_name}")
                        st.write(label)
                        final_state.update(state_update)

                status.update(label="✅ Answer ready!", state="complete", expanded=False)
                result = final_state
            except Exception as exc:
                status.update(label="⚠️ Error encountered", state="error", expanded=True)
                logger.exception("Error executing CRAG graph: %s", exc)
                st.error(
                    f"⚠️ The graph encountered an error: **{type(exc).__name__}** — {exc}\n\n"
                    "Check `rag_assistant.log` for details. Please wait a moment and try again."
                )
                st.stop()

        answer     = result.get("generation", "Sorry, I could not generate an answer.")
        datasource = result.get("datasource", "vectorstore")
        docs       = result.get("documents", [])

        st.write(answer)

        # ── Datasource badge ───────────────────────────────────────────────
        badge = "🌐 Web (Tavily)" if datasource == "web" else f"📚 Document: {current_session.get('filename', 'vectorstore')}"
        st.caption(f"Answered from: **{badge}**")

        # ── Source citations ───────────────────────────────────────────────
        sources = []
        seen    = set()
        fallback_name = current_session.get("filename", "document")

        for doc in docs:
            label = _source_label(doc.metadata, fallback_name)
            if label not in seen:
                seen.add(label)
                sources.append(
                    {
                        "label":   label,
                        "preview": doc.page_content[:150].replace("\n", " ") + "…",
                    }
                )

        if sources:
            with st.expander("📄 Retrieved Context Sources"):
                for src in sources:
                    st.caption(f"**{src['label']}** — {src['preview']}")

    # ── Persist conversation turn into active session ──────────────────────────
    current_session["chat_history"].append(
        {
            "question":   query,
            "answer":     answer,
            "datasource": datasource,
            "sources":    sources,
        }
    )
    session_manager.update_session(current_session)
