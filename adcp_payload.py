"""
AdCP-inspired helpers for emoji detection and payload shaping.
"""

from __future__ import annotations

import re
from typing import Dict


def contains_emoji(text: str) -> bool:
    return bool(re.search(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]", text))


def build_adcp_payload(product: str, audience: str, caption: str, meta: Dict[str, str]) -> Dict:
    """Shape the final output into an AdCP-inspired JSON object."""
    return {
        "adcp_version": "1.0",
        "task": "creative_generation",
        "payload": {
            "target_audience": audience,
            "creative_assets": [
                {
                    "type": "text_ad",
                    "content": caption,
                }
            ],
            "product": product,
        },
        "metadata": {
            "length": int(meta.get("length", len(caption))),
            "word_count": int(meta.get("word_count", len(caption.split()))),
            "sentiment": "energetic",
            "brand_safety_check": meta.get("brand_safety_check", "unknown"),
        },
    }

