# ─────────────────────────────────────────────────────────────────────────────
# Makefile — common developer tasks
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install demo benchmark charts test clean help

PYTHON ?= python3
PROVIDER ?= ollama

help:
	@echo ""
	@echo "  make install    Install Python dependencies"
	@echo "  make demo       Run the interactive tutorial demo (default: Ollama)"
	@echo "  make benchmark  Run the full two-run cost benchmark"
	@echo "  make charts     Generate PNG charts from the latest benchmark results"
	@echo "  make test       Run the test suite"
	@echo "  make clean      Remove generated results and __pycache__"
	@echo ""
	@echo "  Override provider:   make demo PROVIDER=openrouter"
	@echo ""

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

demo:
	$(PYTHON) demo.py --provider $(PROVIDER)

benchmark:
	$(PYTHON) benchmark.py --provider $(PROVIDER)

charts:
	$(PYTHON) charts.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	rm -rf results/__pycache__ tests/__pycache__ __pycache__
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
