# Work Journal CLI Makefile

# Basic commands
.PHONY: help run test-cli init config show test-llm clean docs docs-serve docs-build docs-deploy

help:
	@echo "Work Journal CLI Commands:"
	@echo "  make run          - Launch interactive Terminal UI (MAIN COMMAND)"
	@echo "  make tui          - Launch interactive Terminal UI (alias)"
	@echo "  make test-cli     - Test that CLI is working"
	@echo "  make init         - Initialize configuration"
	@echo "  make config       - Show current configuration"
	@echo "  make test-llm     - Test LLM provider connectivity"
	@echo "  make clean        - Clean up temporary files"
	@echo ""
	@echo "TUI Usage:"
	@echo "  In TUI, use slash commands:"
	@echo "    /add              - Add journal entry for today"
	@echo "    /add --date monday - Add entry for specific date"  
	@echo "    /list             - Browse and select entries"
	@echo "    /quit or Ctrl+Q   - Exit TUI"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies"
	@echo "  make lint         - Run linting (future)"
	@echo "  make test         - Run tests (future)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         - Serve documentation locally (main docs command)"
	@echo "  make docs-serve   - Serve documentation locally (alias)"
	@echo "  make docs-build   - Build documentation site"
	@echo "  make docs-deploy  - Deploy docs to GitHub Pages"

# Main entry point - Launch TUI interface
run:
	uv run python -m src.work_journal.cli tui

# Launch TUI interface (alias)
tui:
	uv run python -m src.work_journal.cli tui

# Hello command for testing
hello:
	uv run python -m src.work_journal.cli hello

# Test that CLI is working
test-cli:
	@echo "Testing CLI help..."
	uv run python -m src.work_journal.cli --help
	@echo ""
	@echo "Testing hello command..."
	uv run python -m src.work_journal.cli hello

# Configuration commands
init:
	uv run python -m src.work_journal.cli config init

config:
	uv run python -m src.work_journal.cli config show

test-llm:
	uv run python -m src.work_journal.cli test-llm

# Development commands
install:
	uv sync

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick commands for different CLI operations
add:
	uv run python -m src.work_journal.cli add --text "Fixed authentication bug with Sarah and Mike"

add-yesterday:
	uv run python -m src.work_journal.cli add --date yesterday --text "Led team meeting about Q4 goals"

# Test date parsing
test-dates:
	@echo "Testing various date formats..."
	uv run python -m src.work_journal.cli add --date "today" --text "Testing today"
	uv run python -m src.work_journal.cli add --date "yesterday" --text "Testing yesterday"
	uv run python -m src.work_journal.cli add --date "monday" --text "Testing last Monday"
	uv run python -m src.work_journal.cli add --date "3 days ago" --text "Testing relative date"

# Documentation commands
docs: docs-serve

docs-serve:
	@echo "Starting documentation server..."
	@echo "Installing mkdocs dependencies..."
	@uv pip install mkdocs-material mkdocs-git-revision-date-localized-plugin
	@echo "Serving docs at http://127.0.0.1:8000"
	@echo "Press Ctrl+C to stop the server"
	uv run mkdocs serve

docs-build:
	@echo "Building documentation site..."
	@uv pip install mkdocs-material mkdocs-git-revision-date-localized-plugin
	uv run mkdocs build --clean
	@echo "Documentation built in ./site/"

docs-deploy:
	@echo "Deploying documentation to GitHub Pages..."
	@uv pip install mkdocs-material mkdocs-git-revision-date-localized-plugin
	uv run mkdocs gh-deploy
	@echo "Documentation deployed to GitHub Pages"

