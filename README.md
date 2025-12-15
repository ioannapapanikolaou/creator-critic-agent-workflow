# The "Critic & Creator" Loop (Agentic Workflow)

This project implements a **two-agent, iterative workflow** (Creator + Critic) for generating short ad copy and returning an **Ad Context Protocol (AdCP)–inspired JSON payload**.

The system uses **LangGraph** for agent orchestration and state management, and can run **entirely locally** using **Ollama**.

---

## Overview

**Input**
- Product name (e.g. `"Neon Energy Drink"`)
- Target audience (e.g. `"Gen-Z Gamers"`)

**Agents**
- **Creator**: proposes short ad captions
- **Critic**: evaluates captions against strict rules

**Rules enforced**
- Caption must mention the product name
- Caption must be **≤ 15 words**
- Caption must include **at least one emoji**
- Caption must avoid blocked safety terms

**Loop**
1. Creator generates a caption
2. Critic evaluates it
3. If rejected, feedback is passed back to the Creator
4. Loop repeats until approved or the attempt budget is exhausted

**Output**
- Attempt-by-attempt evaluation log
- Final approved result formatted as **AdCP-style JSON**

---

## How to Run

### 1) Prerequisites
- Python 3
- Ollama installed and running (for LLM mode)

### 2) Install dependencies
```
pip install -r requirements.txt
```

### 3) Default run (local LLM via Ollama)
```
ollama run llama3:8b   # ensure the model is downloaded and the server is running
python adcp_workflow.py --product "Neon Energy Drink" --audience "Gen-Z Gamers"
```

### 4) Run with API key using OpenAI
```
python adcp_workflow.py \
  --product "Neon Energy Drink" \
  --audience "Gen-Z Gamers" \
  --provider openai \
  --api-key "$OPENAI_API_KEY" \
  --model gpt-4o-mini
``` 

### 5) Template-only mode (fallback - no LLM for testing)
```
python adcp_workflow.py \
  --product "Neon Energy Drink" \
  --audience "Gen-Z Gamers" \
  --creator template
```
## CLI Options

- `--model`: Select a model (default: llama3:8b; used for Ollama or OpenAI)
- `--max-attempts N`: Maximum Creator–Critic iterations (default: 5)
- `--temperature`: Sampling temperature for the Creator LLM (default: 0.4)
- `--ollama-base-url`: Override Ollama server URL (default: http://localhost:11434)
- `--critic`: Choose llm (default) or heuristic rule-based critic
- `--provider`: Choose `ollama` (default, local) or `openai`
- `--api-key`: OpenAI API key (optional if set via env `OPENAI_API_KEY`; ignored for Ollama)
- `--openai-base-url`: OpenAI endpoint

## Output

The script prints:
- An AdCP-style attempt log, showing each proposal, decision, and feedback
- A final AdCP-inspired JSON payload, for example:

```
{
  "adcp_version": "1.0",
  "task": "creative_generation",
  "payload": {
    "target_audience": "Gen-Z Gamers",
    "creative_assets": [
      {
        "type": "text_ad",
        "content": "Neon Energy Drink: power up, Gen-Z Gamers! ⚡"
      }
    ],
    "product": "Neon Energy Drink"
  },
  "metadata": {
    "length": 49,
    "word_count": 7,
    "sentiment": "energetic",
    "brand_safety_check": "passed"
  }
}
```

## Notes & Trade-offs
* LangGraph is used to explicitly model the agent loop and state transitions

(Creator → Critic → retry or terminate).

- Local-first architecture: the default path uses Ollama and requires no cloud APIs.

- LLM + rules hybrid:
    * Creator uses an LLM (or templates as fallback)
    * Critic can be either LLM-based or fully deterministic via heuristic rules
- Determinism vs flexibility:
    * LLM Critic provides flexible judgment
    * Heuristic Critic ensures strict, reproducible enforcement
- Attempt budget prevents infinite loops and makes failure states explicit.

## Transparency on LLM Prompts
- LLMCreator prompt:
    * Generate one caption
    * ≤ 15 words
    * Include product name verbatim
    * Include at least one emoji
    * Avoid blocked safety terms
    * Energetic, concise tone
    * Prior critic feedback is injected to guide retries
- LLMCritic prompt:
    * Strictly validates product mention, word count, emoji presence, and safety terms
    * Responds with APPROVED or REJECTED: <concise feedback>

## Repository Structure
- `agents.py`: Creator and Critic implementations, including LLM-backed variants

- `workflow_graph.py`: LangGraph state definition, routing logic, and retry loop

- `adcp_payload.py`: AdCP-inspired payload builder and emoji detection helper

- `adcp_workflow.py`: CLI entrypoint that runs the workflow and prints JSON output