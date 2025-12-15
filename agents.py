"""
Core agent classes: Creator and Critic, plus shared data structures.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from adcp_payload import contains_emoji


@dataclass
class Evaluation:
    approved: bool
    feedback: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Attempt:
    caption: str
    evaluation: Evaluation


class Creator:
    """Proposes ad copy variations based on feedback signals."""

    def __init__(self, product: str, audience: str) -> None:
        self.product = product
        self.audience = audience
        self.attempts: List[Attempt] = []
        self.templates = [
            "{product}: power up, {audience}! ⚡",
            "Boost with {product} for {audience} 🚀",
            "{audience}, grab {product} and win! 🏆",
            "Stay sharp with {product}, {audience}! ✨",
        ]

    def _pick_template(self) -> str:
        return random.choice(self.templates)

    def _apply_feedback(self, caption: str, feedback: str) -> str:
        """Adjust caption heuristically using feedback keywords."""
        lowered = feedback.lower()

        if "emoji" in lowered and not contains_emoji(caption):
            caption += " 🔥"

        if "too long" in lowered or "under" in lowered:
            words = caption.split()
            caption = " ".join(words[:15])

        if "mention the product" in lowered and self.product.lower() not in caption.lower():
            caption = caption.strip()
            caption = f"{self.product} - {caption}"

        caption = re.sub(r"\s+", " ", caption).strip()
        return caption

    def propose(self, feedback: Optional[str] = None) -> str:
        if feedback is None or not self.attempts:
            caption = self._pick_template().format(
                product=self.product, audience=self.audience
            )
        else:
            last_caption = self.attempts[-1].caption
            caption = self._apply_feedback(last_caption, feedback)
        return caption


class Critic:
    """Evaluates captions against strict acceptance rules."""

    def __init__(self, product: str, max_words: int = 15) -> None:
        self.product = product
        self.max_words = max_words
        self.blocklist = {"kill", "violence", "hate"}

    def evaluate(self, caption: str) -> Evaluation:
        rules_failed: List[str] = []

        if not contains_emoji(caption):
            rules_failed.append("Must contain an emoji.")

        word_count = len(caption.split())
        if word_count > self.max_words:
            rules_failed.append(f"Too long ({word_count} words). Keep under {self.max_words}.")

        if self.product.lower() not in caption.lower():
            rules_failed.append("Please mention the product by name.")

        if any(term in caption.lower() for term in self.blocklist):
            rules_failed.append("Contains blocked terms; rewrite for brand safety.")

        if rules_failed:
            return Evaluation(
                approved=False,
                feedback=" ".join(rules_failed),
                metadata={"status": "rejected"},
            )

        metadata = {
            "status": "approved",
            "length": str(len(caption)),
            "word_count": str(word_count),
            "brand_safety_check": "passed",
        }
        return Evaluation(approved=True, feedback="Approved", metadata=metadata)


class LLMCreator:
    """Creator backed by an injected LangChain chat model (Ollama by default).

    Prompt intent: instructs the model to return ONE caption <=15 words,
    include the product name verbatim, include an emoji, avoid blocked terms, and keep
    an energetic, concise tone. Feedback text is injected to guide retries.
    """

    def __init__(
        self,
        product: str,
        audience: str,
        model: str = "llama3:8b",
        temperature: float = 0.4,
        base_url: str = "http://localhost:11434",
        llm=None,
    ) -> None:
        self.product = product
        self.audience = audience
        if llm is not None:
            self.llm = llm
        else:
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)

    def propose(self, feedback: Optional[str] = None) -> str:
        feedback_text = feedback or "No prior feedback; follow the rules."
        messages = [
            SystemMessage(
                content=(
                    "You are an ad copywriter. Generate ONE short caption that:\n"
                    "- Mentions the product name exactly as given.\n"
                    "- Is 15 words or fewer.\n"
                    "- Includes at least one emoji.\n"
                    "- Avoids blocked terms: kill, violence, hate.\n"
                    "- Tone: energetic, concise."
                )
            ),
            HumanMessage(
                content=(
                    f"Product: {self.product}\n"
                    f"Audience: {self.audience}\n"
                    f"Feedback: {feedback_text}\n"
                    "Return only the caption."
                )
            ),
        ]
        result = self.llm.invoke(messages)
        caption = str(getattr(result, "content", result)).strip()
        # Strip wrapping quotes if the model returns a quoted string.
        caption = caption.strip('"').strip("'")
        caption = re.sub(r"\s+", " ", caption)
        return caption


class LLMCritic:
    """Critic backed by an injected LangChain chat model (Ollama by default).

    Prompt intent (transparency): strict check to ensure the caption mentions the
    product, is <=15 words, includes an emoji, and avoids blocked terms. Responds with
    either APPROVED or REJECTED plus concise violated-rule feedback.
    """

    def __init__(
        self,
        product: str,
        max_words: int = 15,
        model: str = "llama3:8b",
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434",
        llm=None,
    ) -> None:
        self.product = product
        self.max_words = max_words
        if llm is not None:
            self.llm = llm
        else:
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)

    def evaluate(self, caption: str) -> Evaluation:
        messages = [
            SystemMessage(
                content=(
                    "You are a strict ad quality and safety checker.\n"
                    f"- Caption must mention the product name: {self.product}.\n"
                    f"- Caption must be {self.max_words} words or fewer.\n"
                    "- Caption must include at least one emoji.\n"
                    "- Avoid blocked terms: kill, violence, hate.\n"
                    "Respond with either:\n"
                    "APPROVED\n"
                    "or\n"
                    "REJECTED: <concise feedback listing violated rules>"
                )
            ),
            HumanMessage(
                content=(
                    f"Product: {self.product}\n"
                    f"Caption: {caption}\n"
                    "Evaluate now."
                )
            ),
        ]
        result = self.llm.invoke(messages)
        text = str(getattr(result, "content", result)).strip()
        
        first_line = text.strip().splitlines()[0].lower()
        approved = ("approved" in first_line) and ("rejected" not in first_line)

        # approved = text.lower().startswith("approved")

        if approved:
            feedback = "Approved"
            status = "approved"
            brand_safety = "passed"
        else:
            feedback = text.replace("REJECTED:", "").strip() or "Rejected"
            status = "rejected"
            brand_safety = "failed"


        word_count = len(caption.split())
        metadata = {
            "status": status,
            "length": str(len(caption)),
            "word_count": str(word_count),
            "brand_safety_check": brand_safety,
        }
        return Evaluation(approved=approved, feedback=feedback, metadata=metadata)

