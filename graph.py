"""
graph.py
--------
Assembles and compiles the LangGraph CRAG StateGraph.

The compiled `app` object is imported by app.py and invoked per user query.

Graph topology
--------------

START
  └─► query_transform
        └─► [route_question]
              ├─► retrieve ──► grade_documents ──► [decide_to_generate]
              │                                          ├─► generate
              │                                          └─► web_search ──► generate
              └─► web_search ──────────────────────────────────► generate
                                                                    └─► check_hallucination_and_relevance
                                                                              ├─► [useful]        → END
                                                                              ├─► [not_supported] → generate  (retry, max 3)
                                                                              └─► [not_useful]    → web_search → generate
"""

from langgraph.graph import END, START, StateGraph

from state import GraphState
from nodes import (
    check_hallucination_and_relevance,
    generate,
    grade_documents,
    query_transform,
    retrieve,
    web_search,
)
from edges import decide_to_generate, grade_generation, route_question
from logger_config import logger

# ── Build the graph ────────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

# ── Register nodes ─────────────────────────────────────────────────────────────
workflow.add_node("query_transform", query_transform)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node(
    "check_hallucination_and_relevance", check_hallucination_and_relevance
)

# ── Entry point ────────────────────────────────────────────────────────────────
workflow.add_edge(START, "query_transform")

# ── Conditional: route after query_transform ───────────────────────────────────
workflow.add_conditional_edges(
    "query_transform",
    route_question,
    {
        "retrieve": "retrieve",
        "web_search": "web_search",
    },
)

# ── Standard: after retrieve, always grade the documents ──────────────────────
workflow.add_edge("retrieve", "grade_documents")

# ── Conditional: after grade_documents, decide to generate or fall back ────────
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "web_search": "web_search",
    },
)

# ── Standard: after web_search, always generate ───────────────────────────────
workflow.add_edge("web_search", "generate")

# ── Standard: after generate, always self-check ───────────────────────────────
workflow.add_edge("generate", "check_hallucination_and_relevance")

# ── Conditional: after self-check, route to END / retry / web fallback ────────
# The `not_supported` branch needs to increment `retries` before looping.
# We do this by wrapping the edge to inject a state update.

def _increment_retries_and_generate(state: GraphState) -> dict:
    """Thin wrapper: increments `retries` and re-routes to `generate`."""
    current = state.get("retries", 0)
    logger.info("[graph] Incrementing generation retries: %d -> %d", current, current + 1)
    return {"retries": current + 1}


workflow.add_node("increment_retries", _increment_retries_and_generate)
workflow.add_edge("increment_retries", "generate")

workflow.add_conditional_edges(
    "check_hallucination_and_relevance",
    grade_generation,
    {
        "useful": END,
        "not_supported": "increment_retries",
        "not_useful": "web_search",
    },
)

# ── Compile ────────────────────────────────────────────────────────────────────
app = workflow.compile()
