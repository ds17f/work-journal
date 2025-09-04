# Getting Started

Get Work Journal up and running in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/ds17f/work-journal.git
cd work-journal

# Setup with uv (recommended)
uv sync

# Or setup with pip
pip install -e .
```

## First Run

Launch the application:

```bash
# Using make (recommended)
make run

# Or directly with uv
uv run python -m work_journal.cli

# Or if installed with pip
work-journal
```

**First-time users** will be guided through a 4-step onboarding process to set up AI providers and create your first configuration.

## Environment Setup

For cloud providers, set up API keys in environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic  
export ANTHROPIC_API_KEY="your-api-key-here"
```

You can also create a `.env` file in the project directory or use `~/.work-journal/.env` for system-wide configuration.

### Logging Configuration

Control logging behavior with environment variables:

```bash
# Set logging level (DEBUG, INFO, WARNING, ERROR)
export WORK_JOURNAL_LOG_LEVEL=DEBUG

# Suppress console logging (useful for TUI)
export WORK_JOURNAL_SUPPRESS_CONSOLE_LOG=true

# Run with debug logging to troubleshoot LLM issues
WORK_JOURNAL_LOG_LEVEL=DEBUG make run
```

**Log File Location**: `./work-journal.log` in your current working directory

**Log Features**:
- Automatic log rotation (10MB max, 5 backup files)
- Debug information for LLM calls and responses
- Error tracking with detailed context
- LiteLLM API request/response logging at DEBUG level

## File Storage

Work Journal stores all data locally in `~/.work-journal/`:

```
~/.work-journal/
├── entries/           # Daily journal entries (JSON files)
│   ├── 2024-01-15.json
│   └── 2024-01-16.json
├── settings.json      # Provider and configuration settings
├── entities.json      # Collaborators, projects, and tags
└── .env              # API keys (optional)
```

**Key points:**
- All data stays on your machine - no cloud storage
- JSON format makes it easy to backup and export
- Entries are organized by date for easy browsing
- Settings and configurations are portable between machines

## Getting Help

Work Journal includes comprehensive **online help** throughout the application:

- Press **'h'** in any interface to toggle contextual help
- Help content is specific to each screen and function
- All major features have detailed guidance and examples

The application is designed to be self-explanatory with its built-in help system!

## Troubleshooting

### LLM Issues

If you're experiencing problems with AI responses (empty responses, errors, timeouts):

1. **Enable debug logging** to see detailed API information:
   ```bash
   WORK_JOURNAL_LOG_LEVEL=DEBUG make run
   ```

2. **Check the log file** (`./work-journal.log`) for detailed error information

3. **Try a different model** - some models may have availability issues:
   - GPT-4 is generally more stable than GPT-5
   - Local models (Ollama) work offline but may be slower

4. **Verify API keys** and provider configurations in the settings

### Performance Tips

- Use `WORK_JOURNAL_SUPPRESS_CONSOLE_LOG=true` in production to reduce terminal output
- Log files automatically rotate to prevent disk space issues
- Debug logging creates verbose logs - use INFO level for normal operation