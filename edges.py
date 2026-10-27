"""
edges.py
--------
Conditional routing functions for the LangGraph CRAG pipeline.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from state import GraphState
from logger_config import logger

MAX_RETRIES = 1

# Keywords indicating the question is specifically about the user's local/uploaded document
DOCUMENT_KEYWORDS = {
    "resume", "document", "file", "paper", "report", "pdf", "docx",
    "author", "project", "projects", "certification", "certifications",
    "experience", "education", "skill", "skills", "summary", "section",
    "table", "figure", "conclusion", "intro", "introduction", "context",
}


def _is_document_specific_query(query: str) -> bool:
    """Checks if a query is explicitly targeting the uploaded document."""
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
    return bool(tokens & DOCUMENT_KEYWORDS)


class RouteDecision(BaseModel):
    """LLM-structured routing choice."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description=(
            "Choose 'vectorstore' for questions about the uploaded document, resume, text, "
            "analysis, summaries, or domain concepts.\n"
            "Choose 'web_search' ONLY for questions specifically requesting real-time live data, "
            "current news after 2024, or live external search."
        )
    )


def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(model="open-mistral-nemo", temperature=0, max_retries=1, timeout=30)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE 1 — route_question  (after `query_transform`)
# ─────────────────────────────────────────────────────────────────────────────

def route_question(state: GraphState) -> Literal["retrieve", "web_search"]:
    """
    Decides whether to search the local session vector store or web search.
    Always prioritizes the local vector store for document-related queries.
    """
    question = state.get("question", "")

    # Fast-path: If the question explicitly mentions document-specific terms, always retrieve
    if _is_document_specific_query(question):
        logger.info("[route_question] Document-specific keywords detected in '%s' -> 'retrieve'", question)
        return "retrieve"

    logger.info("[route_question] Evaluating routing for query: '%s'", question)

    try:
        llm = _get_llm()
        router = llm.with_structured_output(RouteDecision)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert query router for a document Q&A assistant where a user has uploaded a file.\n"
                    "- Use 'vectorstore' for questions about the document, resume, contents, summaries, or domain topics.\n"
                    "- Use 'web_search' ONLY if the user explicitly asks for live web search, current news, or real-time web info.",
                ),
                ("human", "Question: {question}"),
            ]
        )

        chain = prompt | router
        result: RouteDecision = chain.invoke({"question": question})
        decision = "web_search" if result and result.datasource == "web_search" else "retrieve"
        logger.info("[route_question] Routing decision: '%s' -> %s", question, decision)
        return decision

    except Exception as exc:
        logger.warning("[route_question] Routing error (%s); defaulting to 'retrieve'", exc)
        return "retrieve"


# ─────────────────────────────────────────────────────────────────────────────
# EDGE 2 — decide_to_generate  (after `grade_documents`)
# ─────────────────────────────────────────────────────────────────────────────

def decide_to_generate(
    state: GraphState,
) -> Literal["generate", "web_search"]:
    """
    Determines whether to generate from retrieved chunks or fall back to web search.
    Never falls back to web search if the query is specific to a private document.
    """
    docs = state.get("documents", [])
    web_done = state.get("web_search_attempted", False)
    question = state.get("question", "")

    if docs:
        logger.info("[decide_to_generate] %d relevant chunks available -> 'generate'", len(docs))
        return "generate"

    if web_done:
        logger.info("[decide_to_generate] Web search already attempted -> 'generate'")
        return "generate"

    # If the user is asking about their private file (e.g. resume, document), web search cannot help
    if _is_document_specific_query(question):
        logger.info("[decide_to_generate] 0 chunks found for document query '%s' -> 'generate' (factual not found)", question)
        return "generate"

    logger.info("[decide_to_generate] No relevant local chunks found for general query -> falling back to 'web_search'")
    return "web_search"


# ─────────────────────────────────────────────────────────────────────────────
# EDGE 3 — grade_generation  (after `check_hallucination_and_relevance`)
# ─────────────────────────────────────────────────────────────────────────────

def grade_generation(
    state: GraphState,
) -> Literal["useful", "not_supported", "not_useful"]:
    """
    Evaluates generation grades and terminates cleanly.
    Never falls back to web search for private document queries.
    """
    hallucination = state.get("hallucination_grade", "yes")
    answers_q = state.get("answer_grade", "yes")
    retries = state.get("retries", 0)
    web_done = state.get("web_search_attempted", False) or (state.get("datasource") == "web")
    question = state.get("question", "")
    is_doc_query = _is_document_specific_query(question)

    logger.info(
        "[grade_generation] State: grounded=%s, answers_question=%s, retries=%d, web_attempted=%s, is_doc_query=%s",
        hallucination, answers_q, retries, web_done, is_doc_query
    )

    # 1. Hallucination handling
    if hallucination == "no":
        if retries < MAX_RETRIES:
            logger.info("[grade_generation] Hallucination flagged; retrying generation (%d/%d) -> 'not_supported'",
                        retries + 1, MAX_RETRIES)
            return "not_supported"

        logger.info("[grade_generation] Hallucination retry limit reached.")
        # If it's a private document query, NEVER route to Tavily web search
        if is_doc_query or web_done:
            logger.info("[grade_generation] Document query or web already run; terminating -> 'useful'")
            return "useful"

        logger.info("[grade_generation] General query; falling back to web search -> 'not_useful'")
        return "not_useful"

    # 2. Answers question check
    if answers_q == "yes":
        logger.info("[grade_generation] Answer grounded and addresses question -> 'useful' (END)")
        return "useful"

    # 3. Answer grader flagged off-topic
    if not web_done and not is_doc_query:
        logger.info("[grade_generation] Off-topic general query; trying web search once -> 'not_useful'")
        return "not_useful"

    logger.info("[grade_generation] Terminating -> 'useful' (END)")
    return "useful"
