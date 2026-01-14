"""
CLI entrypoint for the Creator/Critic loop, orchestrated with LangGraph.
"""

from __future__ import annotations

import argparse
import json
import warnings
from typing import List, Dict, Any

from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from adcp_payload import build_adcp_payload
from agents import Attempt
from workflow_graph import run_workflow


def _attempts_as_adcp(attempts: List[Attempt]) -> Dict[str, Any]:
    """Shape attempt history into an AdCP-inspired log object."""
    entries = []
    for idx, attempt in enumerate(attempts, start=1):
        status = attempt.evaluation.metadata.get("status", "unknown")
        entries.append(
            {
                "sequence": idx,
                "type": "text_ad",
                "content": attempt.caption,
                "status": status,
                "feedback": attempt.evaluation.feedback,
            }
        )
    return {
        "adcp_version": "1.0",
        "task": "creative_generation_log",
        "attempts": entries,
    }

# pydantic models used to validate and describe the JSON output 
class AttemptEntryModel(BaseModel):
    sequence: int
    type: str
    content: str
    status: str
    feedback: str

# after generating attempts and final caption, build the attempt log and final payload dicts 
class AttemptLogModel(BaseModel):
    adcp_version: str
    task: str
    attempts: List[AttemptEntryModel]


class CreativeAssetModel(BaseModel):
    type: str
    content: str


class PayloadModel(BaseModel):
    target_audience: str
    creative_assets: List[CreativeAssetModel]
    product: str


class MetadataModel(BaseModel):
    length: int
    word_count: int
    sentiment: str
    brand_safety_check: str


class FinalAdcpModel(BaseModel):
    adcp_version: str
    task: str
    payload: PayloadModel
    metadata: MetadataModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Creator & Critic loop producing AdCP-style JSON output (LangGraph).",
    )
    parser.add_argument("--product", required=True, help="Product name, e.g., 'Neon Energy Drink'")
    parser.add_argument("--audience", required=True, help="Target audience, e.g., 'Gen-Z Gamers'")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Max iterations between creator and critic (default: 5)",
    )
    parser.add_argument(
        "--creator",
        choices=["template", "llm"],
        default="llm",
        help="Use local LLM via Ollama (default) or heuristic templates",
    )
    parser.add_argument(
        "--critic",
        choices=["heuristic", "llm"],
        default="llm",
        help="Use local LLM via Ollama (default) or heuristic rules",
    )
    parser.add_argument(
        "--model",
        default="llama3:8b",
        help="Model name for LLM creator (Ollama or OpenAI). Ignored in template mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature for LLM creator.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama server (provider=ollama).",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM provider. Default is local Ollama; OpenAI requires an API key.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for OpenAI (optional if set via env). Ignored for Ollama.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL (for proxies/self-hosted).",
    )
    args = parser.parse_args()

    final_state = run_workflow(
        args.product,
        args.audience,
        max_attempts=args.max_attempts,
        creator_mode=args.creator,
        critic_mode=args.critic,
        model=args.model,
        temperature=args.temperature,
        base_url=args.ollama_base_url,
        provider=args.provider,
        api_key=args.api_key,
        openai_base_url=args.openai_base_url,
    )

    # # see intermediate messages 
    # print("\nRaw intermediate attempts:")
    # for attempt in final_state["attempts"]:
    #     print(
    #         f"caption='{attempt.caption}', "
    #         f"approved={attempt.evaluation.approved}, "
    #         f"feedback='{attempt.evaluation.feedback}'"
    #     )

    if not final_state.get("approved", False):
        print("\nNo approved caption produced.")
        print("Reason:", final_state.get("feedback", "Unknown"))

    caption = final_state["caption"]
    attempts: List[Attempt] = final_state["attempts"]
    metadata = final_state.get("metadata", {})

    attempts_log = _attempts_as_adcp(attempts)
    AttemptLogModel.model_validate(attempts_log)

    print("\nAttempt Log (AdCP-style):")
    print(json.dumps(attempts_log, indent=2))

    final_payload = build_adcp_payload(
        product=args.product,
        audience=args.audience,
        caption=caption,
        meta=metadata,
    )

    FinalAdcpModel.model_validate(final_payload)

    print("\nFinal AdCP JSON:")
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()

