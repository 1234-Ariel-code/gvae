```makefile
.PHONY: help install install-dev clean lint format test

help:
	@echo "Available commands:"
	@echo "  make install      Install package requirements"
	@echo "  make install-dev  Install package in editable mode"
	@echo "  make clean        Remove Python cache and temporary files"
	@echo "  make lint         Run basic syntax checks"
	@echo "  make format       Format Python files with black if installed"
	@echo "  make test         Run pytest if tests are available"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

lint:
	python -m py_compile gvae/*.py

format:
	@if command -v black >/dev/null 2>&1; then \
		black gvae/*.py; \
	else \
		echo "black is not installed. Install with: pip install black"; \
	fi

test:
	@if command -v pytest >/dev/null 2>&1; then \
		pytest; \
	else \
		echo "pytest is not installed. Install with: pip install pytest"; \
	fi
```
