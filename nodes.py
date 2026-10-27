"""
nodes.py
--------
All node functions for the LangGraph CRAG pipeline.

Each function receives the current `GraphState` dict and returns a
partial dict with only the keys that node modifies.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from state import GraphState
from logger_config import logger

# ── Module-level retriever reference (injected per session by app.py) ─────────
_retriever = None


def set_retriever(retriever) -> None:
    """Call this from app.py to wire the active session's retriever."""
    global _retriever
    _retriever = retriever
    logger.info("[nodes] Active session retriever registered.")


# ── Shared LLM instance ───────────────────────────────────────────────────────
def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(model="open-mistral-nemo", temperature=0, max_retries=1, timeout=30)


# ── Pydantic grading schemas ───────────────────────────────────────────────────

class BatchDocumentGrade(BaseModel):
    """Batch relevance grade: indices of the documents that contain useful information."""
    relevant_indices: list[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of documents that contain information related or useful for "
            "answering the question. Return an empty list ONLY if none are remotely relevant."
        )
    )


class GenerationGrade(BaseModel):
    """Unified evaluation of generated answer."""
    binary_score: Literal["yes", "no"] = Field(
        description=(
            "Is the answer grounded in (supported by) the provided context? "
            "Answer 'yes' if the answer is supported by the context, or truthfully states "
            "that the context lacks the answer. Answer 'no' ONLY if it invents unsupported facts."
        )
    )
    answer_grade: Literal["yes", "no"] = Field(
        description=(
            "Does the answer address the question? "
            "Answer 'yes' if it directly answers or helpfully and accurately explains that the context lacks the answer. "
            "Answer 'no' only if the answer is completely off-topic or irrelevant."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — query_transform
# ─────────────────────────────────────────────────────────────────────────────

def query_transform(state: GraphState) -> dict:
    """
    Rewrites the raw user question into a concise, standalone search query
    that retains conversational context without ballooning prompt tokens.
    """
    raw_question = state.get("question", "").strip()
    history = state.get("chat_history", [])

    if not history:
        logger.info("[query_transform] Turn 1: Keeping original query: '%s'", raw_question)
        return {"question": raw_question}

    logger.info("[query_transform] Follow-up query received: '%s' (%d history turns)", raw_question, len(history))

    # Build concise history (last 2 turns, truncate assistant response)
    history_text = ""
    for turn in history[-2:]:
        user_q = turn.get("question", "")
        asst_a = turn.get("answer", "")
        if len(asst_a) > 200:
            asst_a = asst_a[:200] + "…"
        history_text += f"User: {user_q}\nAssistant: {asst_a}\n"

    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a search query optimizer for a document Q&A assistant.\n"
                    "Given the recent conversation history and a follow-up question, rewrite the follow-up "
                    "into a standalone search query that preserves the specific topic and entities.\n"
                    "If the follow-up question is already clear and self-contained, return it as-is.\n"
                    "Return ONLY the search query — no preamble, no quotes.",
                ),
                (
                    "human",
                    "Conversation history:\n{history}\n\nFollow-up question: {question}",
                ),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        transformed = chain.invoke(
            {"history": history_text, "question": raw_question}
        ).strip().strip('"').strip("'")

        if not transformed:
            transformed = raw_question

        logger.info("[query_transform] Rephrased query: '%s' -> '%s'", raw_question, transformed)
        return {"question": transformed}

    except Exception as exc:
        logger.warning("[query_transform] Transformation failed (%s); using original query.", exc)
        return {"question": raw_question}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — retrieve
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> dict:
    """
    Queries the session-isolated Chroma vector store.
    """
    if _retriever is None:
        logger.warning("[retrieve] No active retriever found. Returning empty documents.")
        return {"documents": [], "datasource": "vectorstore"}

    question = state["question"]
    logger.info("[retrieve] Searching session vector store for: '%s'", question)
    docs = _retriever.invoke(question)
    logger.info("[retrieve] Found %d candidate chunks in active session vector store.", len(docs))
    return {"documents": docs, "datasource": "vectorstore"}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — grade_documents
# ─────────────────────────────────────────────────────────────────────────────

def grade_documents(state: GraphState) -> dict:
    """
    Filters retrieved documents, keeping those relevant to the question.
    """
    docs = state.get("documents", [])
    if not docs:
        logger.info("[grade_documents] No candidate documents to grade.")
        return {"documents": []}

    question = state["question"]
    logger.info("[grade_documents] Grading %d chunks for query: '%s'", len(docs), question)

    try:
        llm = _get_llm()
        grader_llm = llm.with_structured_output(BatchDocumentGrade)

        # Include up to 800 chars per doc chunk
        docs_text = "\n\n".join(
            f"[Document {i}]:\n{doc.page_content[:800]}"
            for i, doc in enumerate(docs)
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a document relevance grader. Given a list of numbered document excerpts "
                    "and a question, return the 0-based indices of all excerpts that contain useful, "
                    "relevant, or contextual information to answer the question.\n"
                    "Be reasonably generous with context that provides relevant facts, background, or entities.\n"
                    "Return an empty list ONLY if the excerpts have absolutely nothing to do with the question.",
                ),
                (
                    "human",
                    "Question: {question}\n\nDocuments:\n{documents}",
                ),
            ]
        )

        result: BatchDocumentGrade = (prompt | grader_llm).invoke(
            {"question": question, "documents": docs_text}
        )

        indices = result.relevant_indices if result and result.relevant_indices else []
        relevant_docs = [
            docs[i] for i in indices
            if 0 <= i < len(docs)
        ]
        logger.info("[grade_documents] %d of %d chunks deemed relevant (indices: %s).",
                    len(relevant_docs), len(docs), indices)
        return {"documents": relevant_docs}

    except Exception as exc:
        logger.warning("[grade_documents] Grading error (%s); retaining all candidate chunks.", exc)
        return {"documents": docs}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — web_search
# ─────────────────────────────────────────────────────────────────────────────

def web_search(state: GraphState) -> dict:
    """
    Tavily web search fallback for external general knowledge queries.
    """
    question = state["question"]
    logger.info("[web_search] Searching Tavily for query: '%s'", question)

    web_docs: list[Document] = []
    try:
        tavily = TavilySearch(max_results=3)
        results = tavily.invoke(question)

        for r in results:
            if isinstance(r, dict):
                content = r.get("content", "") or r.get("text", "")
                url     = r.get("url", "")    or r.get("href", "web")
                title   = r.get("title", "")
            else:
                content = str(r)
                url     = "web"
                title   = ""

            if content:
                web_docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": url, "title": title},
                    )
                )
        logger.info("[web_search] Tavily returned %d web document chunks.", len(web_docs))
    except Exception as exc:
        logger.error("[web_search] Tavily search error: %s", exc)

    merged = state.get("documents", []) + web_docs
    return {
        "documents": merged,
        "datasource": "web",
        "web_search_attempted": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — generate
# ─────────────────────────────────────────────────────────────────────────────

def generate(state: GraphState) -> dict:
    """
    Generates a grounded answer from the current documents in the state.
    """
    question = state["question"]
    docs = state.get("documents", [])
    logger.info("[generate] Generating answer for query: '%s' (from %d document chunks)", question, len(docs))

    if not docs:
        msg = "The provided document does not contain information to answer this question."
        logger.info("[generate] No context documents available. Returning factual 'not found' message.")
        return {"generation": msg}

    context = "\n\n".join(doc.page_content for doc in docs)

    try:
        llm = _get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful, factual, and concise AI assistant for document Q&A.\n"
                    "Answer the user's question using ONLY the provided context.\n"
                    "- If the answer is in the context, provide a clear, factual, and complete answer.\n"
                    "- If the context does not contain the answer, state clearly: "
                    "'The provided document does not contain information about this.'\n"
                    "- Do NOT fabricate information not present in the context.",
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {question}",
                ),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        generation = chain.invoke(
            {"context": context, "question": question}
        )
        logger.info("[generate] Answer generated (%d chars): %s…", len(generation), generation[:100].replace("\n", " "))
        return {"generation": generation}

    except Exception as exc:
        logger.error("[generate] Generation error: %s", exc)
        return {"generation": f"Error generating answer: {exc}. Please try again."}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — check_hallucination_and_relevance
# ─────────────────────────────────────────────────────────────────────────────

def check_hallucination_and_relevance(state: GraphState) -> dict:
    """
    Evaluates whether the generation is grounded and addresses the question.
    Passes up to 2000 chars per document chunk to avoid false hallucination flags.
    """
    docs = state.get("documents", [])
    generation = state.get("generation", "")
    question = state.get("question", "")

    logger.info("[check_hallucination_and_relevance] Evaluating answer against %d chunks", len(docs))

    if not docs:
        return {
            "hallucination_grade": "yes",
            "answer_grade": "yes",
        }

    try:
        llm = _get_llm()
        grader = llm.with_structured_output(GenerationGrade)

        # Allow full text up to 2000 chars per doc to ensure facts aren't prematurely truncated
        doc_text = "\n\n".join(doc.page_content[:2000] for doc in docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an answer evaluator for a RAG pipeline.\n"
                    "Evaluate the generated answer based on the provided documents and question:\n"
                    "1. binary_score: Answer 'yes' if the key facts and entities in the answer are supported by "
                    "or reasonably derived from the documents, or if the answer truthfully states the documents lack the answer. "
                    "Answer 'no' ONLY if the answer invents significant claims or entities completely absent from the context.\n"
                    "2. answer_grade: Answer 'yes' if the answer directly addresses the question or helpfully informs "
                    "the user that the context does not have the answer. Answer 'no' only if it is completely off-topic.",
                ),
                (
                    "human",
                    "Documents:\n{documents}\n\nQuestion: {question}\n\nAnswer: {generation}",
                ),
            ]
        )

        result: GenerationGrade = (prompt | grader).invoke(
            {"documents": doc_text, "question": question, "generation": generation}
        )

        h_grade = result.binary_score if result else "yes"
        a_grade = result.answer_grade if result else "yes"

        logger.info("[check_hallucination_and_relevance] Grade: grounded=%s, answers_question=%s", h_grade, a_grade)
        return {
            "hallucination_grade": h_grade,
            "answer_grade": a_grade,
        }

    except Exception as exc:
        logger.warning("[check_hallucination_and_relevance] Validation call failed (%s); accepting generation.", exc)
        return {
            "hallucination_grade": "yes",
            "answer_grade": "yes",
        }
