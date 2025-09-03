# Provider Configuration

Work Journal supports multiple AI providers. You can use cloud services, local models, or mix both.

## Supported Providers

### Cloud Providers

**OpenAI**
```bash
export OPENAI_API_KEY="your-api-key"
```
Available models: GPT-4, GPT-3.5-turbo, GPT-4-turbo

**Anthropic**  
```bash
export ANTHROPIC_API_KEY="your-api-key"
```
Available models: Claude 3.5 Sonnet, Claude 3.5 Haiku

**Custom OpenAI-Compatible Endpoints**
Any service that implements the OpenAI API format.

### Local Models

**Ollama**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1:8b
```
Runs on `http://localhost:11434/v1` by default.

**LM Studio**
Download and start LM Studio, configure to run on `http://localhost:1234/v1`.

## Model Functions

Work Journal uses three types of model functions:

- **Conversation**: Interactive chat and entry refinement
- **Processing**: Structures raw entries into professional summaries  
- **JIRA Matching**: Finds relevant tickets for your work

You can assign different models to each function during configuration.

## Setup Process

1. **Set up environment variables** for cloud providers (if using)
2. **Install and start local models** (if using)
3. **Run Work Journal** and use the guided setup, or access System menu → Model configuration
4. **Test your configuration** to verify connectivity

## Troubleshooting

**Local Models**: Ensure Ollama or LM Studio is running before testing connections.

**API Keys**: Check your environment variables and `.env` file configuration.

**Connection Issues**: Use System menu → Model configuration → Test connection to verify your setup.

The application includes comprehensive online help - press 'h' in any interface for detailed guidance!