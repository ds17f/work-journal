# Work Journal

A simple, powerful tool to track your work accomplishments using AI-powered entry processing.

## 🚀 Quick Start

```bash
# Install
uv add work-journal

# Run
work-journal
```

The application will guide you through setup on first run. For detailed usage instructions, see the **[User Guide](https://ds17f.github.io/work-journal/)**.

## 🛠️ Developer Setup

### Prerequisites
- Python 3.11+
- `uv` package manager

### Local Development

```bash
# Clone and setup
git clone https://github.com/ds17f/work-journal.git
cd work-journal
uv sync

# Run in development mode
uv run python -m work_journal.cli

# Or install in editable mode
uv pip install -e .
work-journal
```

### Testing & Quality

```bash
# Run tests
uv run pytest

# Format and lint
uv run ruff format
uv run ruff check

# Type checking
uv run mypy src/work_journal

# Build package
uv build
```

### Documentation Development

```bash
# Install docs dependencies
uv add --dev mkdocs-material

# Serve locally
uv run mkdocs serve
# Open http://127.0.0.1:8000

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

## 🏗️ Architecture

**Simple but powerful:**
- **TUI**: Rich-based terminal interface with comprehensive online help
- **Storage**: Local JSON files with soft delete and backup support
- **LLM Integration**: Multi-provider support (OpenAI, Anthropic, local models)
- **Processing**: AI-powered entry structuring and impact assessment
- **Logging**: Comprehensive debug logging with automatic rotation

All data stays local. No telemetry. API keys via environment variables only.

### Debug & Troubleshooting

```bash
# Enable detailed logging for LLM debugging
WORK_JOURNAL_LOG_LEVEL=DEBUG make run

# Check logs (automatically created in working directory)
tail -f work-journal.log
```

## 🤝 Contributing

1. Fork and create feature branch
2. Add tests for new functionality
3. Ensure `uv run pytest && uv run ruff format && uv run ruff check` passes
4. Update docs if needed
5. Use [Conventional Commits](https://www.conventionalcommits.org/)
6. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🆘 Support

- **Documentation**: [User Guide](https://ds17f.github.io/work-journal/)
- **Issues**: [GitHub Issues](https://github.com/ds17f/work-journal/issues)

---

**Built for developers, by developers. Simple, powerful, and designed to make work documentation effortless.**