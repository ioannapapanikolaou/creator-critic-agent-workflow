import pytest

from agents import Evaluation
from workflow_graph import build_graph, _critic_node

# deterministic creator and critics to be able to assert routing behaviour without LLMs

class SequenceCreator:
    def __init__(self, captions):
        self._captions = iter(captions)

    def propose(self, feedback=None):
        try:
            return next(self._captions)
        except StopIteration:
            return "fallback"


class FlakyCritic:
    def __init__(self, approve_on_substring="ok"):
        self.ok = approve_on_substring

    def evaluate(self, caption: str) -> Evaluation:
        if self.ok in caption:
            return Evaluation(
                approved=True,
                feedback="Approved",
                metadata={"status": "approved"},
            )
        return Evaluation(
            approved=False,
            feedback="Needs keyword",
            metadata={"status": "rejected"},
        )


class AlwaysRejectCritic:
    def evaluate(self, caption: str) -> Evaluation:
        return Evaluation(
            approved=False,
            feedback="Always reject",
            metadata={"status": "rejected"},
        )


def test_routing_retries_until_approved():
    creator = SequenceCreator(["bad caption", "this one is ok"])
    critic = FlakyCritic(approve_on_substring="ok")
    app = build_graph(creator, critic, max_attempts=3)

    final_state = app.invoke(
        {
            "product": "Prod",
            "audience": "QA",
            "caption": "",
            "feedback": None,
            "approved": False,
            "attempts": [],
            "metadata": {},
        }
    )

    assert final_state["approved"] is True
    assert len(final_state["attempts"]) == 2
    assert "ok" in final_state["caption"]


def test_attempt_budget_exhaustion_marks_exhausted():
    critic = AlwaysRejectCritic()
    node = _critic_node(critic, max_attempts=0)
    state = {
        "product": "Prod",
        "audience": "QA",
        "caption": "bad caption",
        "feedback": None,
        "approved": False,
        "attempts": [],
        "metadata": {},
        "exhausted": False,
    }
    result = node(state)
    assert result.get("exhausted") is True
    assert result["approved"] is False
    assert "Attempt budget exhausted" in result["feedback"]
