# Getting Started with Work Journal

This guide will help you get Work Journal running with a local AI model in minutes.

## Quick Setup Options

Choose your preferred local AI setup:

## Option 1: LM Studio (Recommended - GUI Interface)

### 1. Install LM Studio
Download from [lmstudio.ai](https://lmstudio.ai/) - available for Mac, Windows, and Linux.

### 2. Download a Model
1. Open LM Studio
2. Go to the "Search" tab
3. Search for and download one of these models:
   - **Llama 3.2 3B Instruct** (fast, ~2GB) - search "llama-3.2-3b-instruct"
   - **Llama 3.1 8B Instruct** (better quality, ~5GB) - search "llama-3.1-8b-instruct"

### 3. Start the Local Server
1. Go to the "Local Server" tab in LM Studio
2. Select your downloaded model
3. Click "Start Server"
4. The server will run on `http://localhost:1234`

## Option 2: Ollama (Command Line)

### 1. Install Ollama
**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download/windows)

### 2. Start Ollama and Install a Model
```bash
# Start Ollama service (keep this running)
ollama serve

# In another terminal, install a fast model (3B parameters)
ollama pull llama3.2:3b

# For better quality (optional)
ollama pull llama3.2:8b

# Verify it's working
ollama list
```

## Install and Run Work Journal

### 1. Install Work Journal
```bash
pip install work-journal
```

### 2. Run Work Journal
```bash
work-journal
```

That's it! Work Journal will default to using LM Studio and automatically fall back to cloud providers if needed.

## What You Get

- **Complete Privacy**: All data stays on your machine
- **No API Keys**: No need for OpenAI or other cloud services
- **Fast Processing**: Local inference is often faster than API calls
- **Always Available**: Works offline
- **Easy Switching**: GUI (LM Studio) or CLI (Ollama) options

## Model Settings

Use the `m` command in Work Journal to:
- Switch between LM Studio and Ollama
- Choose fast (3B) or quality (8B) models
- Test your connections
- Switch to cloud providers if needed

Available presets:
- **lmstudio_fast**: Llama 3.2 3B via LM Studio
- **lmstudio_quality**: Llama 3.1 8B via LM Studio  
- **ollama_fast**: Llama 3.2 3B via Ollama
- **ollama_quality**: Llama 3.2 8B via Ollama
- **cloud_openai**: OpenAI models (needs API key)
- **cloud_anthropic**: Claude models (needs API key)

## Troubleshooting

### LM Studio Issues
**"Connection failed" error:**
1. Make sure LM Studio is running
2. Go to "Local Server" tab and click "Start Server"
3. Verify the server shows as running on port 1234

**Model not loaded:**
1. Go to "Local Server" tab in LM Studio
2. Select your downloaded model from the dropdown
3. Click "Start Server"

### Ollama Issues
**"Connection failed" error:**
```bash
# Make sure Ollama is running
ollama serve

# Make sure the model is installed
ollama list
```

**Model not found:**
```bash
ollama pull llama3.2:3b
```

### Performance Tips
- **LM Studio**: Use the GPU acceleration if available in settings
- **3B models**: Very fast, great for most journal entries
- **8B models**: Higher quality, better for complex entries
- **Switch anytime**: Use the `m` command to change models on the fly

## Cloud Alternatives

If you prefer cloud models, Work Journal also supports:
- OpenAI (set `OPENAI_API_KEY`)
- Anthropic (set `ANTHROPIC_API_KEY`)

Use the `m` command to switch between presets.

## Next Steps

1. Try adding your first entry with `a`
2. Generate a 1:1 summary with `s`
3. Browse your entries with `l`
4. Adjust model settings with `m`

Happy journaling! 🎯