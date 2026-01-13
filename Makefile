PYTHON ?= python
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate

# Runtime defaults (override via make VAR=value)
PRODUCT ?= "Neon Energy Drink"
AUDIENCE ?= "Gen-Z Gamers"
MODEL ?= "llama3:8b"
CREATOR ?= llm
CRITIC ?= llm
PROVIDER ?= ollama
TEMPERATURE ?= 0.4
MAX_ATTEMPTS ?= 5
OLLAMA_BASE_URL ?= http://localhost:11434
OPENAI_BASE_URL ?=
API_KEY ?=
ARGS ?=

.PHONY: help venv install test run run-template run-openai lint clean

help:
	@echo "Targets:"
	@echo "  venv          Create virtual environment at $(VENV)"
	@echo "  install       Install requirements into $(VENV)"
	@echo "  test          Run pytest"
	@echo "  run           Run workflow (defaults: Ollama, llama3:8b)"
	@echo "  run-template  Run template/heuristic mode (no LLM)"
	@echo "  run-openai    Run with OpenAI provider (requires API key)"
	@echo "  clean         Remove venv and pyc"
	@echo ""
	@echo "Override defaults, e.g.:"
	@echo "  make run PRODUCT=\"X\" AUDIENCE=\"Y\" MODEL=\"gpt-4o-mini\" PROVIDER=openai API_KEY=\$$OPENAI_API_KEY"
	@echo "  make run-template PRODUCT=\"X\" AUDIENCE=\"Y\""

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(ACTIVATE) && pip install -r requirements.txt

test:
	$(ACTIVATE) && python -m pytest -q

run:
	$(ACTIVATE) && python adcp_workflow.py \
	  --product '$(PRODUCT)' \
	  --audience '$(AUDIENCE)' \
	  --creator $(CREATOR) \
	  --critic $(CRITIC) \
	  --provider $(PROVIDER) \
	  --model $(MODEL) \
	  --temperature $(TEMPERATURE) \
	  --max-attempts $(MAX_ATTEMPTS) \
	  --ollama-base-url $(OLLAMA_BASE_URL) \
	  $(if $(OPENAI_BASE_URL),--openai-base-url $(OPENAI_BASE_URL)) \
	  $(if $(API_KEY),--api-key $(API_KEY)) \
	  $(ARGS)

run-template:
	$(ACTIVATE) && python adcp_workflow.py \
	  --product '$(PRODUCT)' \
	  --audience '$(AUDIENCE)' \
	  --creator template \
	  --critic heuristic \
	  $(ARGS)

run-openai:
	$(ACTIVATE) && python adcp_workflow.py \
	  --product '$(PRODUCT)' \
	  --audience '$(AUDIENCE)' \
	  --provider openai \
	  --creator $(CREATOR) \
	  --critic $(CRITIC) \
	  --model $(MODEL) \
	  --temperature $(TEMPERATURE) \
	  --max-attempts $(MAX_ATTEMPTS) \
	  $(if $(OPENAI_BASE_URL),--openai-base-url $(OPENAI_BASE_URL)) \
	  $(if $(API_KEY),--api-key $(API_KEY)) \
	  $(ARGS)

lint:
	@echo "No linter configured; add your tool of choice."

clean:
	rm -rf $(VENV) **/__pycache__ **/*.pyc
