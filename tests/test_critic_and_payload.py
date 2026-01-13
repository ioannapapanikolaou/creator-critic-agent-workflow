import pytest

from agents import Critic, Evaluation
from adcp_payload import build_adcp_payload


def test_critic_approves_valid_caption():
    critic = Critic(product="Prod", max_words=10)
    evaluation: Evaluation = critic.evaluate("Prod rocks ⚡")
    assert evaluation.approved is True
    assert evaluation.metadata["brand_safety_check"] == "passed"


def test_critic_rejects_missing_emoji():
    critic = Critic(product="Prod", max_words=10)
    evaluation: Evaluation = critic.evaluate("Prod rocks")
    assert evaluation.approved is False
    assert "emoji" in evaluation.feedback.lower()


def test_payload_shape_and_metadata():
    payload = build_adcp_payload(
        product="Prod",
        audience="QA",
        caption="Prod rocks ⚡",
        meta={"length": "12", "word_count": "3", "brand_safety_check": "passed"},
    )

    assert payload["adcp_version"] == "1.0"
    assert payload["task"] == "creative_generation"
    assert payload["payload"]["product"] == "Prod"
    assert payload["payload"]["target_audience"] == "QA"
    assert payload["payload"]["creative_assets"][0]["type"] == "text_ad"
    assert payload["payload"]["creative_assets"][0]["content"] == "Prod rocks ⚡"
    assert payload["metadata"]["brand_safety_check"] == "passed"
