"""LLM integration with multi-provider support via LiteLLM."""

import os
from typing import Dict, Any, Optional
import litellm
from dotenv import load_dotenv

# Configure litellm to drop unsupported parameters for models like GPT-5
litellm.drop_params = True

# Enable debug logging for litellm
import os
if os.getenv("WORK_JOURNAL_LOG_LEVEL", "INFO").upper() == "DEBUG":
    litellm.set_verbose = True

from .models import Settings, Provider, ModelAssignment
from .storage import Storage
from .logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Simple LLM client using LiteLLM with new Settings model."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.storage = Storage()
        
        # Load environment variables - prefer local .env, fallback to ~/.work-journal/.env
        self._load_env_files()
        
        # Load settings (always returns valid Settings object)
        self.settings = self.storage.load_settings()
    
    def _load_env_files(self):
        """Load environment variables from .env files (local first, then global)."""
        from pathlib import Path
        
        # Try local .env first (current working directory)
        local_env = Path(".env")
        if local_env.exists():
            load_dotenv(local_env)
        
        # Then load global .env (may override or supplement local)
        global_env = self.storage.ensure_env_file()
        load_dotenv(global_env)
    
    def reload_env(self):
        """Reload environment variables from .env files."""
        self._load_env_files()
    
    def save_settings(self):
        """Save current settings to storage."""
        self.storage.save_settings(self.settings)
    
    def _setup_provider(self, provider_name: str) -> Dict[str, Any]:
        """Setup LiteLLM for a specific provider."""
        if provider_name not in self.settings.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
            
        provider = self.settings.providers[provider_name]
        
        setup_kwargs = {
            "api_base": provider.api_base
        }
        
        # Handle authentication
        if provider.auth_env:
            api_key = os.getenv(provider.auth_env)
            if not api_key:
                raise ValueError(f"Environment variable {provider.auth_env} not found. Check your .env file.")
            setup_kwargs["api_key"] = api_key
        elif provider.protocol == "ollama":
            # Ollama doesn't require authentication by default
            setup_kwargs["api_key"] = "ollama"
        elif provider.protocol == "openai_compatible" and not provider.auth_env:
            # Local OpenAI-compatible services (like LM Studio) often don't need real API keys
            setup_kwargs["api_key"] = "not-needed"
        
        return setup_kwargs
    
    def _format_model_name(self, provider_name: str, model: str) -> str:
        """Format model name for LiteLLM based on provider protocol."""
        if provider_name not in self.settings.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
            
        provider = self.settings.providers[provider_name]
        protocol = provider.protocol
        
        if protocol == "ollama":
            return f"ollama/{model}"
        elif protocol == "openai_compatible":
            return f"openai/{model}"
        elif protocol == "anthropic":
            return f"anthropic/{model}"
        else:
            # Fallback to direct model name
            return model
    
    def call_llm(self, need: str, messages: list, **kwargs) -> str:
        """Make an LLM call for a specific need (conversation, processing, jira_matching)."""
        logger.debug(f"Starting LLM call for need: {need}")
        
        # Check if we have a current configuration
        if not self.settings.current_config:
            logger.error("No configuration is currently active")
            raise ValueError("No configuration is currently active")
            
        if self.settings.current_config not in self.settings.configurations:
            logger.error(f"Current configuration '{self.settings.current_config}' not found")
            raise ValueError(f"Current configuration '{self.settings.current_config}' not found")
            
        config = self.settings.configurations[self.settings.current_config]
        logger.debug(f"Using configuration: {self.settings.current_config}")
        
        # Get the model assignment for this need
        if need == "conversation":
            assignment = config.conversation
        elif need == "processing":
            assignment = config.processing  
        elif need == "jira_matching":
            assignment = config.jira_matching
        else:
            raise ValueError(f"Unknown need: {need}. Must be 'conversation', 'processing', or 'jira_matching'")
        
        try:
            # Setup provider
            provider_kwargs = self._setup_provider(assignment.provider)
            
            # Format model name for LiteLLM
            model_name = self._format_model_name(assignment.provider, assignment.model)
            logger.debug(f"Making LLM call to model: {model_name}")
            logger.debug(f"Message count: {len(messages)}, kwargs: {list(kwargs.keys())}")
            
            # Make the call
            response = litellm.completion(
                model=model_name,
                messages=messages,
                **provider_kwargs,
                **kwargs
            )
            
            content = response.choices[0].message.content
            logger.debug(f"LLM response received, length: {len(content) if content else 0}")
            
            return content
            
        except Exception as e:
            logger.error(f"LLM call failed for {need}: {e}")
            raise Exception(f"LLM call failed for {need}: {e}")
    
    def get_available_models(self, provider_name: str) -> list:
        """Get list of available models from a provider."""
        try:
            if provider_name not in self.settings.providers:
                return []
                
            provider = self.settings.providers[provider_name]
            protocol = provider.protocol
            
            if protocol == "openai_compatible":
                # Query OpenAI-compatible endpoint (works for OpenAI, LM Studio, GoCode, etc.)
                import requests
                provider_kwargs = self._setup_provider(provider_name)
                headers = {}
                if "api_key" in provider_kwargs:
                    headers["Authorization"] = f"Bearer {provider_kwargs['api_key']}"
                
                response = requests.get(f"{provider_kwargs['api_base']}/models", headers=headers, timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    return [model["id"] for model in models_data.get("data", [])]
                else:
                    return []
                    
            elif protocol == "ollama":
                # Query Ollama's tags endpoint
                import requests
                # Ollama uses different endpoint structure
                base_url = provider.api_base.replace("/v1", "")  # Remove /v1 for Ollama API
                response = requests.get(f"{base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    return [model["name"] for model in models_data.get("models", [])]
                else:
                    return []
                    
            elif protocol == "anthropic":
                # Return known Anthropic models (Anthropic doesn't have a public models endpoint)
                return ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
                
            else:
                return []
                    
        except Exception as e:
            logger.warning(f"Error getting models for {provider_name}: {e}")
            return []
    
    def test_provider(self, provider_name: str, model: str = None) -> tuple[bool, str]:
        """Test connectivity to a specific provider. Returns (success, error_message)."""
        try:
            if provider_name not in self.settings.providers:
                return False, "Provider not found in settings"
                
            provider_kwargs = self._setup_provider(provider_name)
            
            # Use a test model if not specified
            if model is None:
                # Get available models and use the first one
                available_models = self.get_available_models(provider_name)
                if available_models:
                    model = available_models[0]
                else:
                    model = "gpt-3.5-turbo"  # Fallback
            
            # Format model name for LiteLLM
            formatted_model = self._format_model_name(provider_name, model)
            
            response = litellm.completion(
                model=formatted_model,
                messages=[{"role": "user", "content": "Test message - respond with 'OK'"}],
                max_tokens=10,
                **provider_kwargs
            )
            
            return True, "Connection successful"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Provider test failed for {provider_name}: {error_msg}")
            return False, error_msg