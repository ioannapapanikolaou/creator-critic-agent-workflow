"""
LangGraph-based orchestration for the Creator/Critic loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents import Attempt, Creator, Critic, LLMCreator, LLMCritic

from langchain_ollama import ChatOllama

class GraphState(TypedDict):
    product: str
    audience: str
    caption: str
    feedback: Optional[str]
    approved: bool
    attempts: List[Attempt]
    metadata: Dict[str, str]
    exhausted: bool  

# used for pydantic validation on initial and final state 
class GraphStateModel(BaseModel):
    product: str
    audience: str
    caption: str = ""
    feedback: Optional[str] = None
    approved: bool = False
    attempts: List[Attempt] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)
    exhausted: bool = False

    model_config = {"arbitrary_types_allowed": True}


# creator proposes caption
def _creator_node(creator: Any):
    def node(state: GraphState) -> GraphState:
        # If we already exhausted attempts, don't generate anything new.
        if state.get("exhausted"):
            return state

        caption = creator.propose(state.get("feedback"))
        return {**state, "caption": caption}

    return node

# critic evaluates, appends Attempt, sets feedback/approved/exhausted
def _critic_node(critic: Any, max_attempts: int):
    def node(state: GraphState) -> GraphState:
        # If we already exhausted attempts, just pass through.
        if state.get("exhausted"):
            return state

        evaluation = critic.evaluate(state["caption"])
        # Record this attempt (caption + evaluation) in the history for tracking and exhaustion checks
        attempts = state["attempts"] + [Attempt(caption=state["caption"], evaluation=evaluation)]
        exhausted = (len(attempts) >= max_attempts) and (not evaluation.approved)

        # If exhausted, keep last evaluation + mark exhausted, but DO NOT raise.
        if exhausted:
            return {
                **state,
                "feedback": (
                    f"Attempt budget exhausted after {max_attempts} tries. "
                    f"Last feedback: {evaluation.feedback}"
                ),
                "approved": False,
                "attempts": attempts,
                "metadata": {**evaluation.metadata, "status": "rejected", "exhausted": "true"},
                "exhausted": True,
            }

        return {
            **state,
            "feedback": evaluation.feedback,
            "approved": evaluation.approved,
            "attempts": attempts,
            "metadata": evaluation.metadata,
            "exhausted": False,
        }

    return node

# sends to END if approved or exhausted, else back to creator 
def _route(state: GraphState) -> str:
    if state.get("exhausted"):
        return "done"
    return "done" if state["approved"] else "retry"


def build_graph(creator: Any, critic: Any, max_attempts: int):
    graph = StateGraph(GraphState)
    graph.add_node("creator", _creator_node(creator))
    graph.add_node("critic", _critic_node(critic, max_attempts))

    graph.add_edge("__start__", "creator")
    graph.add_edge("creator", "critic")
    graph.add_conditional_edges("critic", _route, {"done": END, "retry": "creator"})
    return graph.compile()


def run_workflow(
    product: str,
    audience: str,
    max_attempts: int = 5,
    creator_mode: str = "llm",
    critic_mode: str = "llm",
    model: str = "llama3:8b",
    temperature: float = 0.4,
    base_url: str = "http://localhost:11434",
    provider: str = "ollama",
    api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
):
    # IMPORTANT: build separate clients so critic can be deterministic
    creator_llm = _build_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        openai_base_url=openai_base_url,
    )
    critic_llm = _build_llm(
        provider=provider,
        model=model,
        temperature=0.0,
        base_url=base_url,
        api_key=api_key,
        openai_base_url=openai_base_url,
    )

    if creator_mode == "llm":
        creator = LLMCreator(
            product=product,
            audience=audience,
            model=model,
            temperature=temperature,
            base_url=base_url,
            llm=creator_llm,
        )
    else:
        creator = Creator(product=product, audience=audience)

    if critic_mode == "llm":
        critic = LLMCritic(
            product=product,
            max_words=15,
            model=model,
            temperature=0.0,
            base_url=base_url,
            llm=critic_llm,
        )
    else:
        critic = Critic(product=product)

    app = build_graph(creator, critic, max_attempts=max_attempts)

    initial_state: GraphState = {
        "product": product,
        "audience": audience,
        "caption": "",
        "feedback": None,
        "approved": False,
        "attempts": [],
        "metadata": {},
        "exhausted": False,
    }

    # Validate initial state shape
    GraphStateModel.model_validate(initial_state)

    final_state = app.invoke(initial_state)

    # Validate final state shape
    GraphStateModel.model_validate(final_state)
    return final_state


def _build_llm(
    provider: str,
    model: str,
    temperature: float,
    base_url: str,
    api_key: Optional[str],
    openai_base_url: Optional[str],
):
    # Guardrail: prevent accidental ollama model tag with OpenAI provider
    if provider == "openai":
        if ":" in model:
            raise ValueError(
                "provider=openai but model looks like an Ollama tag (e.g. llama3:8b). "
                "Use an OpenAI model name (e.g. gpt-4o-mini, gpt-4.1-mini, etc.)."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=openai_base_url,
        )

    # Default to Ollama local
    return ChatOllama(model=model, temperature=temperature, base_url=base_url)
