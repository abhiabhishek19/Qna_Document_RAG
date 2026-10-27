"""
state.py
--------
Defines the shared GraphState TypedDict that flows through every node
in the LangGraph CRAG (Corrective RAG) pipeline.
"""

from typing import List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """
    Represents the complete mutable state of one CRAG execution pass.

    Fields
    ------
    question : str
        The current user query (may be the rewritten / transformed version
        after `query_transform`).
    chat_history : list
        Previous conversation turns carried in from the Streamlit session so
        that the LLM has multi-turn context when rewriting the query.
    documents : list
        The LangChain `Document` objects that have been retrieved from the
        vector store or fetched from Tavily web search.
    generation : str
        The final (or most recent) answer produced by the `generate` node.
    datasource : str
        Tracks where the answer was ultimately grounded:
        ``'vectorstore'`` or ``'web'``.
    retries : int
        Incremented each time the graph loops back from
        `check_hallucination_and_relevance` to `generate`.  Prevents
        infinite self-correction loops (max 3 retries enforced in edges.py).
    hallucination_grade : str
        Temporary key written by `check_hallucination_and_relevance`:
        ``'yes'`` = grounded, ``'no'`` = hallucinated.
    answer_grade : str
        Temporary key written by `check_hallucination_and_relevance`:
        ``'yes'`` = answers the question, ``'no'`` = off-topic.
    """

    question: str
    chat_history: list
    documents: list
    generation: str
    datasource: str
    retries: int
    web_search_attempted: bool
    hallucination_grade: str   # written by check_hallucination_and_relevance
    answer_grade: str          # written by check_hallucination_and_relevance
