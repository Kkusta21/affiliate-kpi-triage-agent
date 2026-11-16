.PHONY: install test demo api fixtures
install:
	pip install -e ".[dev]"
test:
	pytest
demo:
	python -m triage_agent demo
api:
	uvicorn triage_agent.api:app --reload --port 8000
fixtures:
	python scripts/make_fixtures.py
