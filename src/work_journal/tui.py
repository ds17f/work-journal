"""Simple Terminal User Interface for Work Journal."""

import json
import os
import sys
import tempfile
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from .logging_config import get_logger

logger = get_logger(__name__)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live

from .storage import Storage
from .entry_processor import EntryProcessor
from .date_parser import DateParser
from .models import JournalEntry


class WorkJournalTUI:
    """Simple TUI for Work Journal."""
    
    def __init__(self):
        self.console = Console()
        self.storage = Storage()
        self.running = True
        self.help_visible = False
    
    def _get_current_model_info(self, workflow_step: str = "processing") -> str:
        """Get current model info for display purposes."""
        try:
            from .llm import LLMClient
            llm_client = LLMClient()
            if llm_client.settings.current_config and llm_client.settings.current_config in llm_client.settings.configurations:
                config = llm_client.settings.configurations[llm_client.settings.current_config]
                if workflow_step == "processing" and hasattr(config, 'processing'):
                    return f"{config.processing.provider}/{config.processing.model}"
                elif workflow_step == "conversation" and hasattr(config, 'conversation'):
                    return f"{config.conversation.provider}/{config.conversation.model}"
                elif workflow_step == "jira_matching" and hasattr(config, 'jira_matching'):
                    return f"{config.jira_matching.provider}/{config.jira_matching.model}"
            return "No Config"
        except Exception as e:
            return f"Error: {str(e)[:20]}"
    
    @contextmanager
    def llm_spinner(self, message: str, workflow_step: str = "processing"):
        """Context manager to show an animated spinner during LLM calls."""
        model_info = self._get_current_model_info(workflow_step)
        spinner_text = Text(f"{message} ({model_info})")
        
        spinner = Spinner("dots", text=spinner_text, style="cyan")
        
        with Live(spinner, console=self.console, refresh_per_second=8) as live:
            try:
                yield live
            finally:
                # Spinner automatically stops when context exits
                pass
    
    def _show_detected_entities(self, raw_text: str):
        """Show detected entities from the text to give user transparency."""
        try:
            entity_registry = self.storage.entity_registry
            potential_entities = entity_registry.extract_potential_entities(raw_text)
            matched_entities = entity_registry.match_entities(potential_entities)
            
            # Only show if we found anything interesting
            if not any(potential_entities.values()) and not any(matched_entities.values()):
                return
            
            entity_info = []
            
            # Show potential collaborators
            if potential_entities.get('collaborators'):
                entity_info.append(f"👥 Detected collaborators: {', '.join(potential_entities['collaborators'])}")
            
            # Show matched collaborators with high confidence
            if matched_entities.get('collaborators'):
                high_conf_matches = [m.canonical_name for m in matched_entities['collaborators'] if m.confidence > 0.8]
                if high_conf_matches:
                    entity_info.append(f"👥 Known collaborators: {', '.join(high_conf_matches)}")
            
            # Show potential projects
            if potential_entities.get('projects'):
                entity_info.append(f"📁 Detected projects: {', '.join(potential_entities['projects'])}")
            
            # Show matched projects with high confidence
            if matched_entities.get('projects'):
                high_conf_matches = [m.canonical_name for m in matched_entities['projects'] if m.confidence > 0.8]
                if high_conf_matches:
                    entity_info.append(f"📁 Known projects: {', '.join(high_conf_matches)}")
            
            if entity_info:
                self.console.print(f"\n[dim]🔍 Entity detection:[/dim]")
                for info in entity_info:
                    self.console.print(f"[dim]  {info}[/dim]")
                    
        except Exception as e:
            # Silently fail - entity detection is not critical
            pass
    
    def safe_input(self, prompt: str = "", default: str = None) -> Optional[str]:
        """Safe input that handles Ctrl+C and Ctrl+D gracefully."""
        try:
            result = input(prompt).strip()
            return result if result else default
        except (EOFError, KeyboardInterrupt):
            # Both Ctrl+D and Ctrl+C cancel single-line inputs
            # Print newline for cleaner display after interrupt
            print()
            return None
    
    def show_llm_query(self, title: str, system_prompt: str, user_prompt: str):
        """Display the LLM query for transparency."""
        # Show what we're sending to the LLM
        query_content = f"[bold]System Prompt:[/bold]\n{system_prompt[:200]}{'...' if len(system_prompt) > 200 else ''}\n\n"
        query_content += f"[bold]User Prompt:[/bold]\n{user_prompt[:300]}{'...' if len(user_prompt) > 300 else ''}"
        
        query_panel = Panel(
            query_content,
            title=f"🤖 LLM Query - {title}",
            border_style="cyan",
            expand=False
        )
        self.console.print(query_panel)
    
    def open_editor(self, initial_content: str = "") -> Optional[str]:
        """Open an editor for multi-line text input. Returns the content or None if cancelled."""
        # Determine which editor to use
        editor = os.environ.get('EDITOR', 'nano')  # Default to nano if no EDITOR set
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(initial_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Open editor
            self.console.print(f"[dim]Opening {editor}... (save and exit to continue)[/dim]")
            result = subprocess.run([editor, tmp_file_path], check=True)
            
            # Read the content back
            with open(tmp_file_path, 'r') as f:
                content = f.read().strip()
            
            return content if content else None
            
        except subprocess.CalledProcessError:
            self.console.print(f"[red]Error opening editor {editor}[/red]")
            return None
        except KeyboardInterrupt:
            self.console.print("[yellow]Editor cancelled.[/yellow]")
            return None
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass
    
    def _needs_initial_setup(self) -> bool:
        """Check if the system needs initial configuration setup."""
        from .llm import LLMClient
        
        try:
            llm_client = LLMClient()
            settings = llm_client.settings
            
            # Check if we have any configurations
            if not settings.configurations:
                return True
            
            # Check if we have a current configuration set
            if not settings.current_config:
                return True
                
            # Check if the current configuration exists
            if settings.current_config not in settings.configurations:
                return True
                
            return False
            
        except Exception:
            # If there's any error loading, we need setup
            return True
    
    def _handle_initial_setup(self):
        """Handle initial setup with simple 4-step flow."""
        self.console.print("\n[bold yellow]⚠️  Configuration Required[/bold yellow]")
        self.console.print("The Work Journal needs to be configured with AI models before you can start using it.")
        self.console.print("\n[dim]I'll walk you through 4 simple steps:[/dim]")
        self.console.print("  1. Set up Providers")
        self.console.print("  2. Assign Models") 
        self.console.print("  3. Name Configuration")
        self.console.print("  4. Activate")
        
        self.console.print("\n[green]Let's get you set up![/green]")
        self.console.print("\n[dim]Press Enter to start...[/dim]")
        self.safe_input()
        
        # Run the 4-step configuration builder
        success = self._run_config_builder()
        
        if success:
            self.console.clear()
            self.console.print("\n[bold green]🎉 Setup Complete![/bold green]")
            self.console.print("Your Work Journal is now configured and ready to use!")
            self.console.print("\n[green]Ready to start journaling![/green]")  
            self.console.print("\n[dim]Press Enter to continue to the main menu...[/dim]")
            self.safe_input()
        else:
            self.console.print("\n[yellow]Setup was cancelled. You can try again from the Model Settings menu.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
    
    def _run_config_builder(self) -> bool:
        """Run the complete 4-step configuration builder."""
        from .llm import LLMClient
        from .models import Provider, ModelAssignment, Configuration
        
        llm_client = LLMClient()
        
        # Step 1: Providers
        self.console.clear()
        self.console.print("\n[bold cyan]Step 1: Providers[/bold cyan]")
        self.console.print("Let's set up the AI providers you'll use.")
        
        providers_added = self._step1_add_providers(llm_client)
        if not providers_added:
            return False
            
        # Step 2: Assign Models 
        self.console.clear()
        self.console.print("\n[bold cyan]Step 2: Assign Models[/bold cyan]")
        self.console.print("Now, let's assign models for the 3 needs:")
        self.console.print("  • Conversation - Interactive chat and refinement")
        self.console.print("  • Processing - Structure and analyze work entries")
        self.console.print("  • JIRA Matching - Find relevant JIRA tickets")
        
        assignments = self._step2_assign_models(llm_client)
        if not assignments:
            return False
            
        # Step 3: Name Configuration
        self.console.clear()
        self.console.print("\n[bold cyan]Step 3: Name Configuration[/bold cyan]")
        self.console.print("Give your configuration a name so you can identify it later.")
        
        config_name = self._step3_name_config(llm_client)
        if not config_name:
            return False
            
        # Step 4: Activate 
        self.console.clear()
        self.console.print("\n[bold cyan]Step 4: Activate[/bold cyan]")
        self.console.print(f"Creating and activating configuration '{config_name}'...")
        
        success = self._step4_activate_config(llm_client, config_name, assignments)
        return success
    
    def _step1_add_providers(self, llm_client: 'LLMClient') -> bool:
        """Step 1: Add providers to the system - or use existing ones."""
        # Check if we already have providers
        existing_providers = list(llm_client.settings.providers.keys())
        
        if existing_providers:
            self.console.print(f"\n[green]✅ Found {len(existing_providers)} existing providers:[/green]")
            for provider_name in existing_providers:
                provider = llm_client.settings.providers[provider_name]
                self.console.print(f"  • {provider.service_name}")
            
            self.console.print("\n[bold]Options:[/bold]")
            self.console.print("[green]1.[/green] Use existing providers")
            self.console.print("[green]2.[/green] Add another provider")
            
            while True:
                choice = self.safe_input("\nChoose option (1-2, or 'q' to cancel): ")
                if choice is None:
                    return False
                choice = choice.strip()
                
                if choice.lower() == 'q':
                    return False
                elif choice == '1':
                    return True  # Use existing providers
                elif choice == '2':
                    break  # Continue to add new provider
                else:
                    self.console.print("[red]Invalid choice. Please enter 1 or 2.[/red]")
        
        # Either no existing providers, or user chose to add another
        self.console.print("\nWhat type of AI provider would you like to add?")
        
        provider_options = """
[green]1.[/green] 💻 LM Studio (local, free)
    Run AI models on your computer privately
    
[green]2.[/green] 🦙 Ollama (local, free) 
    Another way to run AI models locally
    
[green]3.[/green] ☁️  OpenAI (cloud, paid)
    GPT-4, GPT-3.5-turbo, etc. - requires API key
    
[green]4.[/green] 🤖 Anthropic (cloud, paid)
    Claude models - requires API key
    
[green]5.[/green] 🔧 Other OpenAI-compatible
    Custom endpoint (LM Studio, local servers, etc.)
"""
        
        self.console.print(provider_options)
        
        while True:
            choice = self.safe_input("\nChoose provider type (1-5, or 'q' to cancel): ")
            if choice is None:  # Handle Ctrl+C/Ctrl+D
                return False
            choice = choice.strip()
            
            if choice.lower() == 'q':
                return False
                
            if choice in ['1', '2', '3', '4', '5']:
                provider = self._create_provider_from_choice(choice)
                if provider:
                    # Test the provider
                    success, error_msg = self._test_provider_connection(llm_client, provider)
                    if success:
                        # Add to settings
                        llm_client.settings.providers[provider.name] = provider
                        llm_client.save_settings()
                        self.console.print(f"\n[green]✅ Provider '{provider.service_name}' added successfully![/green]")
                        self.console.print("\n[dim]Press Enter to continue...[/dim]")
                        self.safe_input()
                        return True
                    else:
                        # Add provider temporarily for debug info
                        llm_client.settings.providers[provider.name] = provider
                        self._show_connection_debug(llm_client, provider.name, error_msg)
                        # Remove it since connection failed
                        if provider.name in llm_client.settings.providers:
                            del llm_client.settings.providers[provider.name]
                        
                        retry = self.safe_input("\nTry again? (y/n): ").strip().lower()
                        if retry != 'y':
                            return False
                        # If retrying, redisplay the provider options
                        self.console.print(provider_options)
                else:
                    return False
            else:
                self.console.print("[red]Invalid choice. Please enter 1-5.[/red]")
    
    def _create_provider_from_choice(self, choice: str) -> 'Provider':
        """Create a provider based on user choice."""
        from .models import Provider
        
        if choice == '1':  # LM Studio
            return self._setup_lmstudio_provider()
        elif choice == '2':  # Ollama
            return self._setup_ollama_provider()
        elif choice == '3':  # OpenAI
            return self._setup_openai_provider()
        elif choice == '4':  # Anthropic
            return self._setup_anthropic_provider()
        elif choice == '5':  # Other
            return self._setup_custom_provider()
        return None
    
    def _setup_lmstudio_provider(self) -> 'Provider':
        """Set up LM Studio provider."""
        from .models import Provider
        
        self.console.print("\n[bold]LM Studio Setup[/bold]")
        self.console.print("Make sure LM Studio is running with a model loaded.")
        self.console.print("Default endpoint: http://localhost:1234/v1")
        
        endpoint = self.safe_input("\nEndpoint URL (or press Enter for default): ")
        if not endpoint:
            endpoint = "http://localhost:1234/v1"
            
        provider_name = self.safe_input("Provider name (or press Enter for 'lmstudio'): ")
        if not provider_name:
            provider_name = "lmstudio"
            
        return Provider(
            name=provider_name,
            service_name="LM Studio",
            protocol="openai_compatible",
            api_base=endpoint
        )
    
    def _setup_ollama_provider(self) -> 'Provider':
        """Set up Ollama provider."""
        from .models import Provider
        
        self.console.print("\n[bold]Ollama Setup[/bold]")
        self.console.print("Make sure Ollama is running with models installed.")
        self.console.print("Default endpoint: http://localhost:11434/v1")
        
        endpoint = self.safe_input("\nEndpoint URL (or press Enter for default): ")
        if not endpoint:
            endpoint = "http://localhost:11434/v1"
            
        provider_name = self.safe_input("Provider name (or press Enter for 'ollama'): ")
        if not provider_name:
            provider_name = "ollama"
            
        return Provider(
            name=provider_name,
            service_name="Ollama",
            protocol="ollama",
            api_base=endpoint
        )
    
    def _setup_openai_provider(self) -> 'Provider':
        """Set up OpenAI provider."""
        from .models import Provider
        
        self.console.print("\n[bold]OpenAI Setup[/bold]")
        self.console.print("You'll need an OpenAI API key from https://platform.openai.com/api-keys")
        self.console.print("The API key should be set in your .env file as OPENAI_API_KEY")
        
        provider_name = self.safe_input("\nProvider name (or press Enter for 'openai'): ")
        if not provider_name:
            provider_name = "openai"
            
        return Provider(
            name=provider_name,
            service_name="OpenAI",
            protocol="openai_compatible", 
            api_base="https://api.openai.com/v1",
            auth_env="OPENAI_API_KEY"
        )
    
    def _setup_anthropic_provider(self) -> 'Provider':
        """Set up Anthropic provider."""
        from .models import Provider
        
        self.console.print("\n[bold]Anthropic Setup[/bold]")
        self.console.print("You'll need an Anthropic API key from https://console.anthropic.com/")
        self.console.print("The API key should be set in your .env file as ANTHROPIC_API_KEY")
        
        provider_name = self.safe_input("\nProvider name (or press Enter for 'anthropic'): ")
        if not provider_name:
            provider_name = "anthropic"
            
        return Provider(
            name=provider_name,
            service_name="Anthropic",
            protocol="anthropic",
            api_base="https://api.anthropic.com",
            auth_env="ANTHROPIC_API_KEY"
        )
    
    def _setup_custom_provider(self) -> 'Provider':
        """Set up custom OpenAI-compatible provider."""
        from .models import Provider
        
        self.console.print("\n[bold]Custom Provider Setup[/bold]")
        self.console.print("This is for OpenAI-compatible endpoints like local servers, etc.")
        
        service_name = self.safe_input("\nService name (e.g., 'My Local Server'): ")
        if not service_name:
            self.console.print("[red]Service name is required.[/red]")
            return None
            
        endpoint = self.safe_input("Endpoint URL (e.g., 'http://localhost:8000/v1'): ")
        if not endpoint:
            self.console.print("[red]Endpoint URL is required.[/red]")
            return None
            
        provider_name = self.safe_input("Provider name (e.g., 'my_server'): ")
        if not provider_name:
            provider_name = service_name.lower().replace(' ', '_')
            
        auth_env = self.safe_input("Environment variable for API key (optional, press Enter to skip): ")
        
        return Provider(
            name=provider_name,
            service_name=service_name,
            protocol="openai_compatible",
            api_base=endpoint,
            auth_env=auth_env if auth_env else None
        )
    
    def _test_provider_connection(self, llm_client: 'LLMClient', provider: 'Provider') -> tuple[bool, str]:
        """Test connection to a provider. Returns (success, error_message)."""
        self.console.print(f"\n[yellow]Testing connection to {provider.service_name}...[/yellow]")
        
        # Temporarily add provider to test it
        provider_was_present = provider.name in llm_client.settings.providers
        original_provider = None
        if provider_was_present:
            original_provider = llm_client.settings.providers[provider.name]
        
        llm_client.settings.providers[provider.name] = provider
        
        try:
            success, error_msg = llm_client.test_provider(provider.name)
            if success:
                self.console.print(f"[green]✅ Connection successful![/green]")
            else:
                self.console.print(f"[red]❌ Connection failed: {error_msg}[/red]")
            return success, error_msg
        finally:
            # Restore original state
            if provider_was_present:
                llm_client.settings.providers[provider.name] = original_provider
            else:
                del llm_client.settings.providers[provider.name]
    
    def _show_connection_debug(self, llm_client: 'LLMClient', provider_name: str, error_msg: str = None):
        """Show debug info for failed connection."""
        if provider_name not in llm_client.settings.providers:
            return
            
        provider = llm_client.settings.providers[provider_name]
        
        self.console.print(f"\n[bold blue]🔍 Debug Info:[/bold blue]")
        if error_msg:
            self.console.print(f"[bold red]Error: {error_msg}[/bold red]")
        self.console.print(f"Endpoint: {provider.api_base}")
        self.console.print(f"Protocol: {provider.protocol}")
        
        if provider.auth_env:
            import os
            env_value = os.getenv(provider.auth_env)
            if env_value:
                masked_value = env_value[:8] + "..." if len(env_value) > 8 else env_value
                self.console.print(f"Auth: [green]✅ {provider.auth_env}={masked_value}[/green]")
            else:
                self.console.print(f"Auth: [red]❌ {provider.auth_env} not found in environment[/red]")
                self.console.print(f"[dim]Add this to your .env file: {provider.auth_env}=your_api_key_here[/dim]")
        else:
            self.console.print("Auth: [yellow]None required[/yellow]")
            
        # Common issues and suggestions
        self.console.print(f"\n[bold blue]💡 Common Issues:[/bold blue]")
        
        # Local service specific suggestions
        if provider.protocol == "openai_compatible" and ("localhost" in provider.api_base or "127.0.0.1" in provider.api_base):
            if "1234" in provider.api_base:  # LM Studio default port
                self.console.print(f"[yellow]• Make sure LM Studio is running with a model loaded[/yellow]")
                self.console.print(f"[yellow]• Check that LM Studio's server is started (green icon)[/yellow]")
            elif "11434" in provider.api_base:  # Ollama default port  
                self.console.print(f"[yellow]• Make sure Ollama is running: ollama serve[/yellow]")
                self.console.print(f"[yellow]• Check that you have models installed: ollama list[/yellow]")
            else:
                self.console.print(f"[yellow]• Make sure your local server is running[/yellow]")
            
            if not provider.api_base.endswith('/v1'):
                self.console.print(f"[yellow]• Try adding /v1 to endpoint: {provider.api_base}/v1[/yellow]")
        else:
            # Remote service suggestions
            if not provider.api_base.endswith('/v1'):
                self.console.print(f"[yellow]• Try adding /v1 to endpoint: {provider.api_base}/v1[/yellow]")
            if provider.auth_env and not os.getenv(provider.auth_env):
                self.console.print(f"[yellow]• Missing API key in environment variable {provider.auth_env}[/yellow]")
            self.console.print(f"[yellow]• Verify network connectivity to {provider.api_base}[/yellow]")
    
    def _step2_assign_models(self, llm_client: 'LLMClient') -> dict:
        """Step 2: Assign models for the 3 needs."""
        if not llm_client.settings.providers:
            self.console.print("[red]No providers available. Please add a provider first.[/red]")
            return None
            
        assignments = {}
        needs = [
            ("conversation", "💬 Conversation", "Interactive chat and refinement"),
            ("processing", "⚙️  Processing", "Structure and analyze work entries"),
            ("jira_matching", "🎫 JIRA Matching", "Find relevant JIRA tickets")
        ]
        
        for need_key, need_title, need_desc in needs:
            self.console.clear()
            self.console.print(f"\n[bold cyan]Assign Model for {need_title}[/bold cyan]")
            self.console.print(f"{need_desc}")
            
            assignment = self._select_provider_and_model(llm_client, need_title)
            if not assignment:
                return None
                
            assignments[need_key] = assignment
            
        return assignments
    
    def _select_provider_and_model(self, llm_client: 'LLMClient', need_title: str) -> 'ModelAssignment':
        """Select provider and model for a specific need."""
        from .models import ModelAssignment
        
        # Show available providers
        providers = list(llm_client.settings.providers.keys())
        
        self.console.print(f"\n[bold]Available Providers:[/bold]")
        for i, provider_name in enumerate(providers, 1):
            provider = llm_client.settings.providers[provider_name]
            self.console.print(f"  {i}. {provider.service_name}")
            
        while True:
            choice = self.safe_input(f"\nChoose provider for {need_title} (1-{len(providers)}, or 'q' to cancel): ").strip()
            
            if choice.lower() == 'q':
                return None
                
            if choice.isdigit() and 1 <= int(choice) <= len(providers):
                selected_provider = providers[int(choice) - 1]
                break
            else:
                self.console.print(f"[red]Invalid choice. Please enter 1-{len(providers)}.[/red]")
        
        # Get available models from selected provider
        self.console.print(f"\n[yellow]Getting models from {llm_client.settings.providers[selected_provider].service_name}...[/yellow]")
        
        with self.console.status("Fetching models..."):
            available_models = llm_client.get_available_models(selected_provider)
        
        if not available_models:
            self.console.print(f"[red]❌ No models available from {selected_provider}[/red]")
            return None
            
        self.console.print(f"[green]✅ Found {len(available_models)} models[/green]")
        
        # Show models with pagination
        selected_model = self._select_model_with_pagination(available_models, need_title)
        if not selected_model:
            return None
            
        return ModelAssignment(provider=selected_provider, model=selected_model)
    
    def _select_model_with_pagination(self, models: list, need_title: str) -> str:
        """Select a model from a paginated list."""
        models_per_page = 15
        current_page = 0
        max_pages = (len(models) - 1) // models_per_page + 1
        
        while True:
            self.console.clear()
            self.console.print(f"\n[bold cyan]Select Model for {need_title}[/bold cyan]")
            
            # Show current page of models in a box
            start_idx = current_page * models_per_page
            end_idx = min(start_idx + models_per_page, len(models))
            
            self.console.print(f"\n┌─ [bold]Available Models (Page {current_page + 1} of {max_pages})[/bold] " + "─" * (60 - len(f"Available Models (Page {current_page + 1} of {max_pages})")) + "┐")
            for i in range(start_idx, end_idx):
                model_num = i + 1
                self.console.print(f"│  {model_num:2d}. {models[i]:<50} │")
            self.console.print("└" + "─" * 62 + "┘")
            
            # Show navigation options in a box
            self.console.print(f"\n┌─ [bold]Options[/bold] " + "─" * 53 + "┐")
            if max_pages > 1:
                if current_page > 0:
                    self.console.print("│  [green]p[/green] - Previous page" + " " * 42 + " │")
                if current_page < max_pages - 1:
                    self.console.print("│  [green]n[/green] - Next page" + " " * 46 + " │")
            
            self.console.print(f"│  [green]1-{len(models)}[/green] - Select model by number" + " " * (35 - len(str(len(models)))) + " │")
            self.console.print("│  [red]q[/red] - Cancel" + " " * 48 + " │")
            self.console.print("└" + "─" * 62 + "┘")
            
            choice = self.safe_input(f"\nChoose model for {need_title}: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'n' and current_page < max_pages - 1:
                current_page += 1
                continue
            elif choice == 'p' and current_page > 0:
                current_page -= 1
                continue
            elif choice.isdigit():
                model_num = int(choice)
                if 1 <= model_num <= len(models):
                    selected_model = models[model_num - 1]
                    self.console.print(f"\n[green]✅ Selected: {selected_model}[/green]")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    self.safe_input()
                    return selected_model
                else:
                    self.console.print(f"[red]Invalid choice. Please enter 1-{len(models)}[/red]")
                    self.console.print("\n[dim]Press Enter to try again...[/dim]")
                    self.safe_input()
            else:
                self.console.print("[red]Invalid choice. Try again.[/red]")
                self.console.print("\n[dim]Press Enter to try again...[/dim]")
                self.safe_input()
    
    def _step3_name_config(self, llm_client: 'LLMClient') -> str:
        """Step 3: Name the configuration."""
        # Suggest a name based on providers used
        provider_names = list(llm_client.settings.providers.keys())
        suggested_name = "_".join(provider_names[:2])  # Use first 2 provider names
        
        self.console.print(f"\n[dim]Examples: 'work_setup', 'local_models', 'openai_main'[/dim]")
        self.console.print(f"[dim]Suggested: {suggested_name}[/dim]")
        
        while True:
            config_name = self.safe_input(f"\nConfiguration name (or press Enter for '{suggested_name}'): ").strip()
            
            if not config_name:
                config_name = suggested_name
            
            if config_name.lower() == 'q':
                return None
                
            # Validate name
            if not config_name.replace('_', '').replace('-', '').isalnum():
                self.console.print("[red]Please use only letters, numbers, underscores, and hyphens.[/red]")
                continue
                
            if config_name in llm_client.settings.configurations:
                self.console.print(f"[red]Configuration '{config_name}' already exists. Choose a different name.[/red]")
                continue
                
            return config_name
    
    def _step4_activate_config(self, llm_client: 'LLMClient', config_name: str, assignments: dict) -> bool:
        """Step 4: Create and activate the configuration."""
        from .models import Configuration
        
        try:
            # Create the configuration
            config = Configuration(
                name=config_name,
                conversation=assignments["conversation"],
                processing=assignments["processing"], 
                jira_matching=assignments["jira_matching"]
            )
            
            # Add to settings
            llm_client.settings.configurations[config_name] = config
            llm_client.settings.current_config = config_name
            
            # Save settings
            llm_client.save_settings()
            
            self.console.print(f"\n[green]✅ Configuration '{config_name}' created and activated![/green]")
            
            # Show summary
            self.console.print("\n[bold]Configuration Summary:[/bold]")
            self.console.print(f"  💬 Conversation: {assignments['conversation'].provider} / {assignments['conversation'].model}")
            self.console.print(f"  ⚙️  Processing: {assignments['processing'].provider} / {assignments['processing'].model}")
            self.console.print(f"  🎫 JIRA Matching: {assignments['jira_matching'].provider} / {assignments['jira_matching'].model}")
            
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Error creating configuration: {e}[/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
    
    def _simplified_provider_setup(self, llm_client: 'LLMClient') -> str:
        """Simplified provider selection for onboarding."""
        self.console.clear()
        self.console.print("\n[bold cyan]Step 1: Choose Your AI Provider[/bold cyan]")
        
        self.console.print("What AI provider would you like to use?")
        provider_options = """
[green]1.[/green] 💻 LM Studio (local, free)
    Run AI models on your computer privately
    
[green]2.[/green] 🦙 Ollama (local, free) 
    Another way to run AI models locally
    
[green]3.[/green] ☁️  OpenAI (cloud, paid)
    GPT-4, GPT-3.5-turbo, etc. - requires API key
    
[green]4.[/green] 🤖 Anthropic (cloud, paid)
    Claude models - requires API key
    
[green]5.[/green] 🏢 Other OpenAI-compatible service
    Custom endpoint with OpenAI-compatible API
        """
        
        self.console.print(provider_options)
        self.console.print("\n[cyan bold]Provider choice (1-5) or 'q' to cancel>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if choice is None or choice.lower() == 'q':
            return None
        
        from .models import ProviderConfig
        config = llm_client.config
        
        if choice == '1':
            # LM Studio
            provider_name = "lmstudio_local"
            provider_config = ProviderConfig(
                protocol="openai_compatible",
                service_name="LM Studio (Local)",
                api_base="http://localhost:1234/v1",
                api_key="lm-studio"
            )
            config.providers[provider_name] = provider_config
            
            self.console.print("\n[green]✅ LM Studio configured![/green]")
            self.console.print("Using endpoint: http://localhost:1234/v1")
            self.console.print("\n[yellow]Make sure LM Studio is running before testing.[/yellow]")
            
        elif choice == '2':
            # Ollama
            provider_name = "ollama_local"
            provider_config = ProviderConfig(
                protocol="ollama",
                service_name="Ollama (Local)",
                api_base="http://localhost:11434/v1"
            )
            config.providers[provider_name] = provider_config
            
            self.console.print("\n[green]✅ Ollama configured![/green]")
            self.console.print("Using endpoint: http://localhost:11434/v1")
            self.console.print("\n[yellow]Make sure Ollama is running before testing.[/yellow]")
            
        elif choice == '3':
            # OpenAI
            provider_name = "openai_cloud"
            provider_config = ProviderConfig(
                protocol="openai_compatible",
                service_name="OpenAI (Cloud)",
                api_base="https://api.openai.com/v1",
                auth_env="OPENAI_API_KEY"
            )
            config.providers[provider_name] = provider_config
            
            self.console.print("\n[green]✅ OpenAI configured![/green]")
            self.console.print("Using endpoint: https://api.openai.com/v1")
            self.console.print("\n[yellow]You need to add your API key to the .env file:[/yellow]")
            self.console.print("  OPENAI_API_KEY=sk-your-key-here")
            
        elif choice == '4':
            # Anthropic
            provider_name = "anthropic_cloud"
            provider_config = ProviderConfig(
                protocol="anthropic",
                service_name="Anthropic (Cloud)",
                api_base="https://api.anthropic.com",
                auth_env="ANTHROPIC_API_KEY"
            )
            config.providers[provider_name] = provider_config
            
            self.console.print("\n[green]✅ Anthropic configured![/green]")
            self.console.print("Using endpoint: https://api.anthropic.com")
            self.console.print("\n[yellow]You need to add your API key to the .env file:[/yellow]")
            self.console.print("  ANTHROPIC_API_KEY=your-key-here")
            
        elif choice == '5':
            # Other OpenAI-compatible
            self.console.print("\n[bold]Custom OpenAI-Compatible Service[/bold]")
            self.console.print("Enter the API endpoint URL:")
            self.console.print("[cyan bold]Endpoint>[/cyan bold] ", end="")
            endpoint = self.safe_input()
            
            if not endpoint:
                self.console.print("[red]No endpoint provided. Setup cancelled.[/red]")
                return None
            
            self.console.print("Enter a name for this service:")
            self.console.print("[cyan bold]Service name>[/cyan bold] ", end="")
            service_name = self.safe_input()
            
            if not service_name:
                service_name = "Custom Service"
            
            self.console.print("Enter the environment variable name for your API key:")
            self.console.print("[cyan bold]Env var name>[/cyan bold] ", end="")
            env_var = self.safe_input()
            
            if not env_var:
                self.console.print("[red]No environment variable provided. Setup cancelled.[/red]")
                return None
            
            provider_name = "custom_openai"
            provider_config = ProviderConfig(
                protocol="openai_compatible",
                service_name=service_name,
                api_base=endpoint,
                auth_env=env_var  
            )
            config.providers[provider_name] = provider_config
            
            self.console.print(f"\n[green]✅ {service_name} configured![/green]")
            self.console.print(f"Using endpoint: {endpoint}")
            self.console.print(f"\n[yellow]You need to add your API key to the .env file:[/yellow]")
            self.console.print(f"  {env_var}=your-key-here")
            
        else:
            self.console.print("[red]Invalid choice. Please select 1-5.[/red]")
            return None
        
        return provider_name
    
    def _environment_setup_loop(self, llm_client: 'LLMClient', provider_name: str) -> bool:
        """Environment setup loop with testing and retry."""
        provider_config = llm_client.config.providers[provider_name]
        
        # If no auth required (like LM Studio, Ollama), test immediately
        if not provider_config.auth_env and not provider_config.api_key:
            return self._test_provider_by_name(llm_client, provider_name)
        
        # If auth required, guide through .env setup
        while True:
            self.console.clear()
            self.console.print("\n[bold cyan]Step 2: Environment Setup[/bold cyan]")
            
            from pathlib import Path
            local_env = Path(".env")
            global_env = llm_client.storage.base_path / ".env"
            
            self.console.print(f"[bold]Environment Files:[/bold]")
            self.console.print(f"  Local .env: {local_env.absolute()} {'✅' if local_env.exists() else '❌'}")
            self.console.print(f"  Global .env: {global_env} {'✅' if global_env.exists() else '❌'}")
            
            if provider_config.auth_env:
                self.console.print(f"\n[yellow]You need to add your API key to one of these .env files:[/yellow]")
                self.console.print(f"  {provider_config.auth_env}=your-api-key-here")
                self.console.print(f"\n[dim]I recommend using the local .env file in your project directory.[/dim]")
            
            self.console.print(f"\n[green]Options:[/green]")
            self.console.print(f"  [cyan]t[/cyan] - Test connection (after adding API key)")
            self.console.print(f"  [cyan]e[/cyan] - Show environment debug info")
            self.console.print(f"  [cyan]c[/cyan] - Continue anyway (skip testing)")
            self.console.print(f"  [cyan]q[/cyan] - Quit setup")
            
            self.console.print(f"\n[cyan bold]Choice>[/cyan bold] ", end="")
            choice = self.safe_input()
            
            if choice is None or choice.lower() == 'q':
                return False
            elif choice.lower() == 'c':
                return True  # Continue without testing
            elif choice.lower() == 'e':
                self._show_environment_debug(llm_client, provider_name)
            elif choice.lower() == 't':
                if self._test_provider_by_name(llm_client, provider_name):
                    self.console.print("\n[bold green]🎉 Connection successful![/bold green]")
                    self.console.print("Your provider is working correctly.")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    self.safe_input()
                    return True
                else:
                    self.console.print("\n[red]❌ Connection failed.[/red]")
                    self.console.print("Please check your API key and try again.")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    self.safe_input()
            else:
                self.console.print("[red]Invalid choice. Please select t, e, c, or q.[/red]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
    
    def _test_provider_by_name(self, llm_client: 'LLMClient', provider_name: str) -> bool:
        """Test connection to a specific provider by name."""
        try:
            # Reload environment variables first
            llm_client.reload_env()
            
            # Get available models as a connection test
            models = llm_client.get_available_models(provider_name)
            return len(models) > 0
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def _show_environment_debug(self, llm_client: 'LLMClient', provider_name: str):
        """Show environment debug info for a specific provider."""
        self.console.clear()
        self.console.print("\n[bold blue]🔍 Environment Debug Info[/bold blue]")
        
        # Reload environment first
        llm_client.reload_env()
        
        provider_config = llm_client.config.providers[provider_name]
        
        self.console.print(f"\n[bold]Provider: {provider_config.service_name}[/bold]")
        self.console.print(f"Protocol: {provider_config.protocol}")
        self.console.print(f"Endpoint: {provider_config.api_base}")
        
        if provider_config.auth_env:
            import os
            env_value = os.getenv(provider_config.auth_env)
            if env_value:
                # Show only first few chars for security
                masked_value = env_value[:8] + "..." if len(env_value) > 8 else env_value
                self.console.print(f"Environment Variable: [green]✅ {provider_config.auth_env}={masked_value}[/green]")
            else:
                self.console.print(f"Environment Variable: [red]❌ {provider_config.auth_env} not found[/red]")
        elif provider_config.api_key:
            masked_key = provider_config.api_key[:8] + "..." if len(provider_config.api_key) > 8 else provider_config.api_key
            self.console.print(f"API Key: [green]✅ {masked_key}[/green]")
        else:
            self.console.print("Authentication: [yellow]None required[/yellow]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _simple_model_assignment(self, llm_client: 'LLMClient', provider_name: str) -> bool:
        """Simple model assignment for onboarding - Step 3."""
        self.console.clear()
        self.console.print("\n[bold cyan]Step 3: Assign Models[/bold cyan]")
        
        provider_config = llm_client.config.providers[provider_name]
        self.console.print(f"Getting available models from {provider_config.service_name}...")
        
        # Get available models
        with self.console.status("Fetching models..."):
            available_models = llm_client.get_available_models(provider_name)
        
        if not available_models:
            self.console.print(f"[red]❌ Could not get models from {provider_config.service_name}[/red]")
            self.console.print("This might mean:")
            self.console.print("  • The service is not running (for local providers)")
            self.console.print("  • Authentication is not working")
            self.console.print("  • The endpoint is not correct")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        self.console.print(f"[green]✅ Found {len(available_models)} models[/green]")
        
        # Workflows to configure
        workflows = [
            ("conversation", "💬 Conversation Processing", "Used for interactive chat and refinement"),
            ("processing", "⚙️  Entry Processing", "Used to structure and analyze your work entries"),
            ("jira_matching", "🎫 JIRA Matching", "Used to find relevant JIRA tickets (optional)")
        ]
        
        workflow_configs = {}
        
        for workflow_key, workflow_title, workflow_desc in workflows:
            self.console.clear()
            self.console.print(f"\n[bold cyan]Step 3: Assign Models - {workflow_title}[/bold cyan]")
            self.console.print(f"{workflow_desc}\n")
            
            # Show available models with pagination
            models_per_page = 15
            current_page = 0
            max_pages = (len(available_models) - 1) // models_per_page + 1
            
            while True:
                self.console.print(f"[bold]Available Models (Page {current_page + 1} of {max_pages}):[/bold]")
                start_idx = current_page * models_per_page
                end_idx = min(start_idx + models_per_page, len(available_models))
                
                for i in range(start_idx, end_idx):
                    model_num = i + 1
                    self.console.print(f"  {model_num:2d}. {available_models[i]}")
                
                self.console.print("\n[bold]Options:[/bold]")
                if max_pages > 1:
                    if current_page > 0:
                        self.console.print("  [green]p[/green] - Previous page")
                    if current_page < max_pages - 1:
                        self.console.print("  [green]n[/green] - Next page")
                
                self.console.print("  [green]1-{max_num}[/green] - Select model by number".format(max_num=len(available_models)))
                self.console.print("  [yellow]s[/yellow] - Skip this workflow")
                self.console.print("  [red]q[/red] - Cancel setup")
                
                choice = self.safe_input(f"\nChoose model for {workflow_title}: ").strip().lower()
                
                if choice == 'q':
                    return False
                elif choice == 's':
                    self.console.print(f"[yellow]Skipping {workflow_title}[/yellow]")
                    break
                elif choice == 'n' and current_page < max_pages - 1:
                    current_page += 1
                    continue
                elif choice == 'p' and current_page > 0:
                    current_page -= 1
                    continue
                elif choice.isdigit():
                    model_num = int(choice)
                    if 1 <= model_num <= len(available_models):
                        selected_model = available_models[model_num - 1]
                        
                        from .models import WorkflowSettings
                        workflow_configs[workflow_key] = WorkflowSettings(
                            provider=provider_name,
                            model=selected_model
                        )
                        
                        self.console.print(f"[green]✅ Selected: {selected_model}[/green]")
                        self.console.print("\n[dim]Press Enter to continue...[/dim]")
                        self.safe_input()
                        break
                    else:
                        self.console.print(f"[red]Invalid choice. Please enter 1-{len(available_models)}[/red]")
                else:
                    self.console.print("[red]Invalid choice. Try again.[/red]")
        
        if not workflow_configs:
            self.console.print("[yellow]No workflows configured. Setup cancelled.[/yellow]")
            return False
        
        # Set up the workflows
        llm_client.config.workflows = workflow_configs
        
        # Save the configuration
        llm_client.storage.save_config(llm_client.config)
        
        self.console.clear()
        self.console.print(f"\n[green]✅ Configuration saved![/green]")
        self.console.print(f"Configured workflows using {provider_config.service_name}:")
        
        for workflow_key, config in workflow_configs.items():
            workflow_name = {
                "conversation": "💬 Conversation",
                "processing": "⚙️  Processing", 
                "jira_matching": "🎫 JIRA Matching"
            }.get(workflow_key, workflow_key)
            self.console.print(f"  {workflow_name}: {config.model}")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
        
        return True
    
    def _name_configuration(self, llm_client: 'LLMClient') -> str:
        """Ask user to name their configuration and save it as a preset."""
        self.console.clear()
        self.console.print("\n[bold cyan]Step 4: Name Your Configuration[/bold cyan]")
        self.console.print("Give your configuration a name so you can identify it later.")
        
        # Suggest a name based on the provider(s) used
        provider_names = []
        for workflow_config in llm_client.config.workflows.values():
            provider_config = llm_client.config.providers.get(workflow_config.provider)
            if provider_config and provider_config.service_name not in provider_names:
                provider_names.append(provider_config.service_name)
        
        suggested_name = "_".join(provider_names).lower().replace(" ", "_")
        if not suggested_name:
            suggested_name = "my_config"
            
        self.console.print(f"\n[dim]Examples: 'ollama_local', 'openai_cloud', 'work_setup'[/dim]")
        self.console.print(f"[dim]Suggested: {suggested_name}[/dim]")
        
        while True:
            config_name = self.safe_input(f"\nConfiguration name (or press Enter for '{suggested_name}'): ").strip()
            
            if not config_name:
                config_name = suggested_name
            
            if config_name.lower() == 'q':
                return None
                
            # Validate name
            if not config_name.replace('_', '').replace('-', '').isalnum():
                self.console.print("[red]Please use only letters, numbers, underscores, and hyphens.[/red]")
                continue
                
            if config_name in llm_client.config.presets:
                self.console.print(f"[red]Configuration '{config_name}' already exists. Choose a different name.[/red]")
                continue
                
            break
        
        # Save the current workflows as a preset
        from .models import WorkflowSettings
        preset_workflows = {}
        for workflow_key, workflow_config in llm_client.config.workflows.items():
            preset_workflows[workflow_key] = workflow_config
        
        llm_client.config.presets[config_name] = preset_workflows
        llm_client.config.current_preset = config_name
        
        # Save the configuration
        llm_client.storage.save_config(llm_client.config)
        
        self.console.print(f"\n[green]✅ Configuration '{config_name}' saved and activated![/green]")
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
        
        return config_name
    
    def _handle_name_existing_config(self):
        """Handle naming an existing unnamed configuration."""
        from .llm import LLMClient
        
        self.console.print("\n[bold yellow]🏷️  Configuration Found[/bold yellow]")
        self.console.print("You have an existing configuration that needs a name.")
        self.console.print("This will help you identify it in the future.")
        
        llm_client = LLMClient()
        
        # Show what's currently configured
        self.console.print("\n[bold]Current Setup:[/bold]")
        for workflow_key, workflow_config in llm_client.config.workflows.items():
            provider_config = llm_client.config.providers.get(workflow_config.provider, {})
            service_name = getattr(provider_config, 'service_name', workflow_config.provider)
            
            workflow_name = {
                "conversation": "💬 Conversation",
                "processing": "⚙️  Processing", 
                "jira_matching": "🎫 JIRA Matching"
            }.get(workflow_key, workflow_key)
            
            self.console.print(f"  {workflow_name}: {service_name} / {workflow_config.model}")
        
        config_name = self._name_configuration(llm_client)
        
        if config_name:
            self.console.clear()
            self.console.print(f"\n[bold green]✅ Configuration '{config_name}' saved![/bold green]")
            self.console.print("Your existing setup has been preserved and named.")
        else:
            self.console.print("\n[yellow]Configuration not named - you can do this later in Model Settings.[/yellow]")
            
        self.console.print("\n[dim]Press Enter to continue to the main menu...[/dim]")
        self.safe_input()
    
    def _run_add_provider_flow(self, llm_client: 'LLMClient') -> bool:
        """Run the simplified 3-step setup flow to add a new provider."""
        self.console.clear()
        self.console.print("\n[bold cyan]Add New Provider[/bold cyan]")
        self.console.print("This will walk you through adding a new AI provider to your configuration.")
        self.console.print("You can set up additional providers and assign them to different workflows.")
        
        self.console.print("\n[dim]Press Enter to continue or 'q' to cancel...[/dim]")
        choice = self.safe_input()
        if choice and choice.lower() == 'q':
            return False
        
        # Step 1: Simplified provider selection
        provider_name = self._simplified_provider_setup(llm_client)
        if not provider_name:
            self.console.print("\n[yellow]Provider setup cancelled.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        # Step 2: Environment setup loop with testing
        success = self._environment_setup_loop(llm_client, provider_name)
        if not success:
            self.console.print("\n[yellow]Environment setup incomplete.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
            
        # Step 3: Model assignment (but show existing workflows and let user choose what to update)
        config_updated = self._add_provider_model_assignment(llm_client, provider_name)
        if not config_updated:
            self.console.print("\n[yellow]Model assignment cancelled.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        # Success!
        self.console.clear()
        self.console.print("\n[bold green]🎉 Provider Added Successfully![/bold green]")
        provider_config = llm_client.config.providers[provider_name]
        self.console.print(f"Added: {provider_config.service_name}")
        self.console.print("\nYou can now use this provider for different workflows or create new configurations.")
        
        self.console.print("\n[green]Provider successfully added to your configuration![/green]")  
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
        return True
    
    def _add_provider_model_assignment(self, llm_client: 'LLMClient', provider_name: str) -> bool:
        """Model assignment for adding a provider - shows existing workflows and allows updates."""
        self.console.clear()
        self.console.print("\n[bold cyan]Step 3: Assign Models for New Provider[/bold cyan]")
        
        provider_config = llm_client.config.providers[provider_name]
        self.console.print(f"Getting available models from {provider_config.service_name}...")
        
        # Get available models
        with self.console.status("Fetching models..."):
            available_models = llm_client.get_available_models(provider_name)
        
        if not available_models:
            self.console.print(f"[red]❌ Could not get models from {provider_config.service_name}[/red]")
            self.console.print("This might mean:")
            self.console.print("  • The service is not running (for local providers)")
            self.console.print("  • Authentication is not working")
            self.console.print("  • The endpoint is not correct")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        self.console.print(f"[green]✅ Found {len(available_models)} models[/green]")
        
        # Show current workflow configuration
        self.console.print("\n[bold]Current Workflow Configuration:[/bold]")
        current_workflows = llm_client.config.workflows
        
        workflows_info = [
            ("conversation", "💬 Conversation Processing", "Used for interactive chat and refinement"),
            ("processing", "⚙️  Entry Processing", "Used to structure and analyze your work entries"),
            ("jira_matching", "🎫 JIRA Matching", "Used to find relevant JIRA tickets")
        ]
        
        for workflow_key, workflow_title, workflow_desc in workflows_info:
            if workflow_key in current_workflows:
                current = current_workflows[workflow_key]
                current_provider_config = llm_client.config.providers.get(current.provider, {})
                current_service_name = getattr(current_provider_config, 'service_name', current.provider)
                self.console.print(f"  {workflow_title}: {current_service_name} / {current.model}")
            else:
                self.console.print(f"  {workflow_title}: [red]Not configured[/red]")
        
        self.console.print(f"\n[yellow]Which workflows would you like to update to use {provider_config.service_name}?[/yellow]")
        self.console.print("You can choose to update existing workflows or skip to keep current settings.")
        
        updated_workflows = {}
        
        for workflow_key, workflow_title, workflow_desc in workflows_info:
            self.console.clear()
            self.console.print(f"\n[bold cyan]Update {workflow_title}?[/bold cyan]")
            self.console.print(f"{workflow_desc}\n")
            
            if workflow_key in current_workflows:
                current = current_workflows[workflow_key]
                current_provider_config = llm_client.config.providers.get(current.provider, {})
                current_service_name = getattr(current_provider_config, 'service_name', current.provider)
                self.console.print(f"[bold]Currently using:[/bold] {current_service_name} / {current.model}")
            else:
                self.console.print("[bold]Currently:[/bold] [red]Not configured[/red]")
            
            self.console.print(f"\n[bold]Available options:[/bold]")
            self.console.print(f"  [green]y[/green] - Update to use {provider_config.service_name}")
            self.console.print(f"  [yellow]n[/yellow] - Keep current setting")
            self.console.print(f"  [red]q[/red] - Cancel and go back")
            
            choice = self.safe_input(f"\nUpdate {workflow_title} to use {provider_config.service_name}? (y/n/q): ").strip().lower()
            
            if choice == 'q':
                return False
            elif choice == 'y':
                # Show model selection for this workflow
                selected_model = self._select_model_from_list(available_models, workflow_title)
                if selected_model:
                    from .models import WorkflowSettings
                    updated_workflows[workflow_key] = WorkflowSettings(
                        provider=provider_name,
                        model=selected_model
                    )
                    self.console.print(f"[green]✅ {workflow_title} will use: {selected_model}[/green]")
                else:
                    self.console.print(f"[yellow]Skipped {workflow_title}[/yellow]")
            else:
                self.console.print(f"[yellow]Keeping current setting for {workflow_title}[/yellow]")
                
            if choice != 'q':
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
        
        # Apply the updates
        if updated_workflows:
            # Update the workflows
            for workflow_key, new_config in updated_workflows.items():
                llm_client.config.workflows[workflow_key] = new_config
            
            # Save the configuration
            llm_client.storage.save_config(llm_client.config)
            
            self.console.clear()
            self.console.print(f"\n[green]✅ Configuration updated![/green]")
            self.console.print(f"Updated workflows to use {provider_config.service_name}:")
            
            for workflow_key, config in updated_workflows.items():
                workflow_name = {
                    "conversation": "💬 Conversation",
                    "processing": "⚙️  Processing", 
                    "jira_matching": "🎫 JIRA Matching"
                }.get(workflow_key, workflow_key)
                self.console.print(f"  {workflow_name}: {config.model}")
            
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return True
        else:
            self.console.print(f"\n[yellow]No workflows were updated. The provider {provider_config.service_name} has been added but is not assigned to any workflows.[/yellow]")
            self.console.print("You can assign it to workflows later through the Model Settings menu.")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return True
    
    def _select_model_from_list(self, available_models: list, workflow_title: str) -> str:
        """Helper method to select a model from a paginated list."""
        models_per_page = 15
        current_page = 0
        max_pages = (len(available_models) - 1) // models_per_page + 1
        
        while True:
            self.console.clear()
            self.console.print(f"\n[bold cyan]Select Model for {workflow_title}[/bold cyan]")
            
            self.console.print(f"[bold]Available Models (Page {current_page + 1} of {max_pages}):[/bold]")
            start_idx = current_page * models_per_page
            end_idx = min(start_idx + models_per_page, len(available_models))
            
            for i in range(start_idx, end_idx):
                model_num = i + 1
                self.console.print(f"  {model_num:2d}. {available_models[i]}")
            
            self.console.print("\n[bold]Options:[/bold]")
            if max_pages > 1:
                if current_page > 0:
                    self.console.print("  [green]p[/green] - Previous page")
                if current_page < max_pages - 1:
                    self.console.print("  [green]n[/green] - Next page")
            
            self.console.print("  [green]1-{max_num}[/green] - Select model by number".format(max_num=len(available_models)))
            self.console.print("  [yellow]s[/yellow] - Skip this workflow")
            self.console.print("  [red]q[/red] - Cancel")
            
            choice = self.safe_input(f"\nChoose model for {workflow_title}: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 's':
                return None
            elif choice == 'n' and current_page < max_pages - 1:
                current_page += 1
                continue
            elif choice == 'p' and current_page > 0:
                current_page -= 1
                continue
            elif choice.isdigit():
                model_num = int(choice)
                if 1 <= model_num <= len(available_models):
                    selected_model = available_models[model_num - 1]
                    self.console.print(f"[green]✅ Selected: {selected_model}[/green]")
                    return selected_model
                else:
                    self.console.print(f"[red]Invalid choice. Please enter 1-{len(available_models)}[/red]")
                    self.console.print("\n[dim]Press Enter to try again...[/dim]")
                    self.safe_input()
            else:
                self.console.print("[red]Invalid choice. Try again.[/red]")
                self.console.print("\n[dim]Press Enter to try again...[/dim]")
                self.safe_input()
        
    def run(self):
        """Run the main TUI loop."""
        self.console.clear()
        self.show_welcome()
        
        # Check if configuration is set up
        if self._needs_initial_setup():
            self._handle_initial_setup()
        
        first_loop = True
        
        while self.running:
            try:
                # Clear screen when returning to main menu (but not on first load)
                if not first_loop:
                    self.console.clear()
                first_loop = False
                
                # Show status and menu
                self.show_status()
                self.show_main_menu()
                
                # Get command using basic input
                self.console.print("\n[green bold]>[/green bold] ", end="")
                try:
                    command = input().strip().lower()
                except KeyboardInterrupt:
                    # Ctrl+C at main prompt - just redraw menu (don't exit)
                    self.console.print("\n[dim]Use 'q' to quit.[/dim]")
                    continue
                except EOFError:
                    # Ctrl+D at main prompt - graceful exit
                    self.console.print("\n[dim]Goodbye! 👋[/dim]")
                    self.quit()
                    break
                
                if command:
                    try:
                        self.handle_main_command(command)
                    except KeyboardInterrupt:
                        # Handle Ctrl+C from within menu functions
                        self.console.print("\n[dim]Menu interrupted. Returning to main menu.[/dim]")
                        continue
                        
            except KeyboardInterrupt:
                # This should rarely happen - emergency exit
                self.quit()
                break
    
    def show_welcome(self):
        """Show welcome screen."""
        welcome_text = """
[bold cyan]📝 Work Journal[/bold cyan]

A simple tool to track your work accomplishments.

Use single-letter commands to navigate:
        """
        
        self.console.print(Panel(welcome_text, title="Welcome", border_style="cyan"))
    
    def show_main_menu(self):
        """Show main menu options."""
        # Show help if toggled on
        if self.help_visible:
            help_text = """
[bold]Daily Workflow:[/bold]
[green]a[/green] - Add accomplishments for any date
    Write what you did, AI extracts key details and impact

[green]l[/green] - Browse your work history  
    Paginated list with search and detailed entry views

[green]s[/green] - Create manager-ready summaries
    Generate Top 3 highlights for 1:1 meetings, exported as markdown (has help)

[green]x[/green] - Access settings and data management
    Model configuration, entity management, backups (has help)

[bold]Navigation:[/bold]
[green]h[/green] - Toggle this help  •  [green]c[/green] - Clear screen  •  [green]q[/green] - Quit

[bold]Entry Tips:[/bold]
• Write multiple detailed lines describing your work
• Use natural dates: 'yesterday', 'last friday', '2024-01-15'
• Ctrl+D to finish entry, Ctrl+C to cancel anytime
            """
            self.console.print(Panel(help_text.strip(), title="Help", border_style="blue"))
            self.console.print()
        
        help_text = "Help (hide)" if self.help_visible else "Help (show)"
        menu_text = f"""
  [green]a[/green] - Add new journal entry
  [green]l[/green] - List and browse entries  
  [green]s[/green] - Generate 1:1 summary
  [green]x[/green] - System menu
  [green]h[/green] - {help_text}
        """
        
        self.console.print(menu_text)
    
    def show_status(self):
        """Show current status."""
        recent_entries = self.storage.load_recent_entries(7)
        today = datetime.now().strftime("%A, %B %d, %Y")
        
        # Get current model info
        try:
            from .llm import LLMClient
            llm_client = LLMClient()
            if llm_client.settings.current_config and llm_client.settings.current_config in llm_client.settings.configurations:
                config = llm_client.settings.configurations[llm_client.settings.current_config]
                model_info = f"🤖 {llm_client.settings.current_config} ({config.processing.provider}/{config.processing.model})"
            else:
                model_info = "🤖 No configuration active"
        except:
            model_info = "🤖 model loading..."
        
        status_text = f"📅 {today}  📊 {len(recent_entries)} entries this week  {model_info}"
        self.console.print(f"\n[dim]{status_text}[/dim]")
    
    def handle_main_command(self, command: str):
        """Handle main menu commands."""
        if command == 'a':
            self.add_entry()
        elif command == 'l':
            self.browse_entries()
        elif command == 's':
            self.generate_summary()
        elif command == 'x':
            self.system_menu()
        elif command == 'h':
            self.help_visible = not self.help_visible
        elif command == 'c':
            self.clear_screen()
        elif command == 'q':
            self.quit()
        elif command == '':
            # Empty command, just redraw menu
            pass
        else:
            self.console.print(f"[red]Unknown command: '{command}'. Use single letters from the menu.[/red]")
    
    def add_entry(self):
        """Add a new journal entry."""
        self.console.clear()
        # Get date from user with retry loop
        self.console.print("\n[bold]📝 Add Journal Entry[/bold]")
        
        # Show help if toggled on from main menu
        if self.help_visible:
            help_text = """
[bold]Adding Journal Entries:[/bold]

[green]Date Input:[/green]
• Use natural language: 'today', 'yesterday', 'last friday'
• Specific dates: '2024-01-15', 'jan 15', 'last monday'
• Type 'today' explicitly or use other date expressions
• Ctrl+C or Ctrl+D to cancel and return to main menu

[green]Input Methods:[/green]
• [cyan]1.[/cyan] Type directly - Multi-line input, press Ctrl+D when done
• [cyan]2.[/cyan] Open editor - Uses your $EDITOR (vim, nano, code, etc.)

[green]Writing Tips:[/green]
• Be specific about what you accomplished
• Include impact, outcomes, and collaboration details
• Multiple detailed lines work better than brief summaries
• AI will structure and enhance your raw notes

[green]Controls:[/green]
• Ctrl+D - Finish entry and process with AI
• Ctrl+C - Cancel entry at any time
• Type 'cancel' - Alternative way to cancel during text input
            """
            self.console.print(Panel(help_text.strip(), title="Entry Help", border_style="blue"))
            self.console.print()
        
        date = None
        while date is None:
            self.console.print("[dim]What date is this entry for? (e.g., 'today', 'yesterday', 'last monday', '2025-08-28'):[/dim]")
            self.console.print("[green]Date>[/green] ", end="")
            date_input = self.safe_input()
            if date_input is None:
                self.console.print("\n[yellow]Entry cancelled.[/yellow]")
                return
            
            if not date_input:
                date_input = "today"
                
            # Try to parse date
            try:
                date = self.parse_date_with_llm(date_input)
                date_display = DateParser.format_date_display(date)
                self.console.print(f"\n[bold]📝 Adding entry for {date_display}[/bold]")
            except ValueError as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
                self.console.print("[yellow]Please try again.[/yellow]\n")
        
        # Ask how they want to input the text
        self.console.print("\n[bold]How would you like to write your entry?[/bold]")
        self.console.print("  [cyan]1.[/cyan] Type directly here (multi-line)")
        self.console.print("  [cyan]2.[/cyan] Open in editor ($EDITOR)")
        self.console.print("  [cyan]q.[/cyan] Cancel")
        
        self.console.print("\n[green]Input method>[/green] ", end="")
        choice = self.safe_input()
        if choice is None or choice.lower() == 'q':
            self.console.print("\n[yellow]Entry cancelled.[/yellow]")
            return
        
        raw_text = None
        
        if choice == '1':
            # Direct input method
            self.console.print("\n[dim]Enter your accomplishments (press Ctrl+D when done, Ctrl+C to cancel):[/dim]")
            
            # Collect multi-line input
            lines = []
            try:
                while True:
                    try:
                        line = input()
                        if line.strip().lower() == 'cancel':
                            self.console.print("[yellow]Entry cancelled.[/yellow]")
                            return
                        lines.append(line)
                    except KeyboardInterrupt:
                        # Ctrl+C pressed - cancel entry
                        self.console.print("\n[yellow]Entry cancelled.[/yellow]")
                        return
                    except EOFError:
                        # Ctrl+D pressed - end of input (finish)
                        if lines and any(line.strip() for line in lines):
                            # We have content, proceed with processing
                            break
                        else:
                            # No content, treat as cancel
                            self.console.print("\n[yellow]Entry cancelled (no content entered).[/yellow]")
                            return
            except KeyboardInterrupt:
                # Ctrl+C pressed - always cancel
                self.console.print("\n[yellow]Entry cancelled.[/yellow]")
                return
            
            raw_text = '\n'.join(lines).strip()
            
        elif choice == '2':
            # Editor method
            raw_text = self.open_editor()
            if raw_text is None:
                self.console.print("[yellow]Entry cancelled.[/yellow]")
                return
        else:
            self.console.print("[red]Invalid choice. Entry cancelled.[/red]")
            return
        
        if not raw_text:
            self.console.print("[yellow]No content entered. Entry cancelled.[/yellow]")
            return
        
        # Interactive processing with LLM
        self.console.print(f"[dim]Starting entry processing for date: {date}[/dim]")
        self.console.print(f"[dim]Text length: {len(raw_text)} characters[/dim]")
        
        try:
            self.interactive_entry_processing(raw_text, date)
        except Exception as e:
            self.console.print(f"[red]❌ Failed to process entry: {e}[/red]")
            self.console.print(f"[dim]Error details: {type(e).__name__}[/dim]")
            
            # Also show the raw text and date for debugging
            self.console.print(f"[dim]Debug - Date: {date}[/dim]")
            self.console.print(f"[dim]Debug - Text preview: {raw_text[:100]}...[/dim]")
            
            self.console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            self.safe_input()
    
    def parse_date_with_llm(self, date_input: str) -> str:
        """Parse natural language dates, with LLM fallback if needed."""
        try:
            # First try the built-in parser
            return DateParser.parse_date(date_input)
        except ValueError as e:
            # If built-in parser fails, try LLM parsing
            try:
                from .llm import LLMClient
                
                llm_client = LLMClient()
                
                today = datetime.now()
                # Calculate last Monday for example
                days_back = (today.weekday() - 0) % 7  # Monday is 0
                if days_back == 0:
                    days_back = 7  # If today is Monday, go to last Monday
                last_monday = today - timedelta(days=days_back)
                
                system_prompt = f"""You are a date parser. Convert natural language dates to YYYY-MM-DD format.
Today is {today.strftime('%Y-%m-%d (%A)')}.

Examples:
- "today" -> {today.strftime('%Y-%m-%d')}
- "yesterday" -> {(today - timedelta(days=1)).strftime('%Y-%m-%d')}
- "last monday" -> {last_monday.strftime('%Y-%m-%d')}
- "2025-08-28" -> 2025-08-28

Respond ONLY with the date in YYYY-MM-DD format."""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Parse this date: {date_input}"}
                ]
                
                # Show date parsing query
                self.show_llm_query("Date Parsing", system_prompt, f"Parse this date: {date_input}")
                
                with self.llm_spinner("🤖 Parsing date", "processing"):
                    response = llm_client.call_llm("processing", messages, max_tokens=20, temperature=0)
                # Extract YYYY-MM-DD from response
                import re
                match = re.search(r'\d{4}-\d{2}-\d{2}', response)
                if match:
                    return match.group(0)
                else:
                    raise ValueError(f"LLM could not parse date: '{date_input}'")
                    
            except Exception as llm_error:
                # If LLM also fails, raise the original error with more context
                raise ValueError(f"Could not parse date '{date_input}'. Please try formats like 'today', 'yesterday', 'last monday', '2025-08-30', or '3 days ago'.")
    
    def interactive_entry_processing(self, raw_text: str, date: str):
        """Interactive refinement of entry processing."""
        # Show potential entities detected in the text
        self._show_detected_entities(raw_text)
        
        try:
            processor = EntryProcessor(storage=self.storage)
            
            # Process entry with spinner
            with self.llm_spinner("🤖 Processing entry", "processing"):
                entry = processor.process_entry(raw_text, date)
            
            if not entry:
                self.console.print("[red]❌ Entry processing returned no result[/red]")
                self.console.print("\n[dim]Press Enter to return to main menu...[/dim]")
                self.safe_input()
                return
            
            while True:
                # Show current processed entry with detailed formatting
                processed = entry.processed
                
                # Header with date and basic info
                date_display = DateParser.format_date_display(entry.date)
                impact_icon = {"individual": "👤", "team": "👥", "organization": "🏢"}.get(processed.impact.scope, "📝")
                
                header_info = f"{impact_icon} {date_display} | Impact: {processed.impact.scope.title()} ({processed.impact.significance}/5)"
                if entry.metadata.get("refinement_count", 0) > 0:
                    header_info += f" | Refined {entry.metadata['refinement_count']}x"
                
                self.console.print(f"\n[bold cyan]📖 PROCESSED ENTRY[/bold cyan]")
                self.console.print(f"[dim]{header_info}[/dim]")
                self.console.print(f"[dim]ID: {str(entry.id)[:8]}... | Created: {entry.timestamp.strftime('%Y-%m-%d %H:%M')}[/dim]\n")
                
                # Main content panels
                summary_panel = Panel(
                    processed.summary,
                    title="📝 Summary",
                    border_style="cyan"
                )
                self.console.print(summary_panel)
                
                work_panel = Panel(
                    processed.work,
                    title="🔧 Detailed Work",
                    border_style="blue"
                )
                self.console.print(work_panel)
                
                # Impact details
                impact_details = f"[bold]Business Value:[/bold] {processed.impact.business_value}"
                if processed.impact.metrics:
                    impact_details += f"\n[bold]Metrics:[/bold]\n" + "\n".join(f"• {metric}" for metric in processed.impact.metrics)
                
                impact_panel = Panel(
                    impact_details,
                    title="📊 Impact Assessment",
                    border_style="green"
                )
                self.console.print(impact_panel)
                
                # Additional info if present
                additional_info = []
                
                if processed.projects:
                    additional_info.append(f"[bold]📁 Projects:[/bold] {', '.join(processed.projects)}")
                
                if processed.collaborators:
                    additional_info.append(f"[bold]👥 Collaborators:[/bold] {', '.join(processed.collaborators)}")
                
                if processed.technical_details:
                    tech_details = "\n".join(f"• {detail}" for detail in processed.technical_details)
                    additional_info.append(f"[bold]🔧 Technical Details:[/bold]\n{tech_details}")
                
                if processed.tags:
                    additional_info.append(f"[bold]🏷️  Tags:[/bold] {', '.join(processed.tags)}")
                
                if additional_info:
                    additional_panel = Panel(
                        "\n\n".join(additional_info),
                        title="📋 Additional Information",
                        border_style="yellow"
                    )
                    self.console.print(additional_panel)
                
                # Ask for feedback
                self.console.print("\n[dim]Options: [green]s[/green]=save, [yellow]r[/yellow]=refine, [red]c[/red]=cancel[/dim]")
                self.console.print("[cyan]Action>[/cyan] ", end="")
                action = self.safe_input()
                if action is None:
                    self.console.print("\n[yellow]Entry cancelled.[/yellow]")
                    break
                action = action.lower()
                
                if action in ['s', '']:
                    # Save entry
                    self.storage.save_entry(entry)
                    self.console.print("[green]✅ Entry saved successfully![/green]")
                    self.console.print("\n[dim]Press Enter to return to main menu...[/dim]")
                    self.safe_input()
                    break
                    
                elif action in ['r']:
                    # Get refinement instructions
                    self.console.print("\n[yellow]What would you like to improve or change?[/yellow]")
                    self.console.print("[dim]Examples: 'Make the summary more impactful', 'Add technical details about Docker', 'Include Sarah as collaborator'[/dim]")
                    self.console.print("[yellow]Refinement>[/yellow] ", end="")
                    refinement = self.safe_input()
                    if refinement is None:
                        self.console.print("\n[yellow]Refinement cancelled.[/yellow]")
                        continue
                    
                    if refinement:
                        # Show refinement query to user
                        system_prompt = """You are helping refine a journal entry based on user feedback. 
Your job is to improve the existing structured entry according to the user's instructions.

Keep the same JSON structure but modify the content based on the feedback.
Be precise and maintain professional tone suitable for promotion reviews."""
                        
                        current_processed = entry.processed
                        import json  # Ensure json is available in local scope  
                        user_prompt = f"""Current entry:
{json.dumps(current_processed.model_dump(), indent=2)}

Original raw input:
{entry.raw_input}

User feedback: {refinement}

Please refine the entry based on this feedback. Respond ONLY with valid JSON in the same format."""
                        
                        self.show_llm_query("Entry Refinement", system_prompt, user_prompt)
                        
                        # Refine with LLM
                        with self.llm_spinner("🤖 Refining entry", "processing"):
                            entry = processor.refine_entry(entry, refinement)
                    
                elif action in ['c']:
                    self.console.print("[yellow]Entry cancelled.[/yellow]")
                    break
                else:
                    self.console.print("[red]Invalid option. Use 's', 'r', or 'c'.[/red]")
                    
        except Exception as e:
            self.console.print(f"[red]❌ Error processing entry: {e}[/red]")
            self.console.print(f"[dim]Error type: {type(e).__name__}[/dim]")
            
            # Show common troubleshooting suggestions
            self.console.print("\n[bold]💡 Common Issues:[/bold]")
            self.console.print("• Check if your AI model configuration is working (try 'x' → 'm' → '3' to test)")
            self.console.print("• Ensure you have a valid internet connection for cloud models")
            self.console.print("• For local models, verify they're running (LM Studio server, Ollama)")
            
            self.console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            self.safe_input()
    
    def browse_entries(self):
        """Browse entries with integrated list and detail view."""
        self.console.clear()
        
        
        entries = self.storage.load_all_entries(50)
        
        if not entries:
            self.console.print("[yellow]No entries found. Use /add to create your first entry![/yellow]")
            return
        
        page_size = 10
        current_page = 0
        total_pages = (len(entries) - 1) // page_size + 1
        mode = "list"  # "list" or "detail"
        detail_index = 0
        
        while True:
            if mode == "list":
                # Show help if toggled on
                if self.help_visible:
                    list_help = """
[bold]Entry List Navigation:[/bold]

[green]Commands:[/green]
• [cyan]1-10[/cyan] - View detailed entry (number shown on left)
• [yellow]p[/yellow] - Previous page (when available)
• [yellow]n[/yellow] - Next page (when available)  
• [yellow]q[/yellow] - Return to main menu

[green]Entry Display:[/green]
• [cyan]Icons[/cyan]: 👤 Individual, 👥 Team, 🏢 Organization impact
• [cyan]Date & Summary[/cyan]: When work was done and brief description
• [cyan]Collaborators[/cyan]: Shows up to 2 people you worked with
• [cyan]Metrics[/cyan]: Key quantifiable results (when available)

[green]Tips:[/green]
• Select any entry number to see full details and editing options
• Entries are sorted by date (newest first)
                    """
                    self.console.print(Panel(list_help.strip(), title="List View Help", border_style="blue"))
                    self.console.print()
                
                # Show list view
                start_idx = current_page * page_size
                end_idx = min(start_idx + page_size, len(entries))
                page_entries = entries[start_idx:end_idx]
                
                self.console.print(f"\n[bold cyan]📋 ENTRIES - Page {current_page + 1}/{total_pages}[/bold cyan]")
                self.console.print(f"[dim]Showing {len(page_entries)} of {len(entries)} entries[/dim]\n")
                
                for i, entry in enumerate(page_entries, 1):
                    processed = entry.processed
                    
                    # Format entry
                    impact_icon = {"individual": "👤", "team": "👥", "organization": "🏢"}.get(processed.impact.scope, "📝")
                    date_display = DateParser.format_date_display(entry.date)
                    
                    entry_text = f"  [cyan]{i:2d}.[/cyan] {impact_icon} {date_display}: {processed.summary}"
                    
                    # Add collaborators if any
                    if processed.collaborators:
                        collab_list = ', '.join(processed.collaborators[:2])
                        if len(processed.collaborators) > 2:
                            collab_list += f" +{len(processed.collaborators)-2} more"
                        entry_text += f"\n      [dim]👥 {collab_list}[/dim]"
                    
                    # Add impact metrics if any
                    if processed.impact.metrics:
                        entry_text += f"\n      [dim]📊 {processed.impact.metrics[0]}[/dim]"
                    
                    self.console.print(f"{entry_text}\n")
                
                # Show navigation options
                nav_options = []
                if current_page > 0:
                    nav_options.append("[yellow]p[/yellow]=prev page")
                if current_page < total_pages - 1:
                    nav_options.append("[yellow]n[/yellow]=next page")
                nav_options.append("[yellow]q[/yellow]=quit")
                
                self.console.print(f"[dim]Options: 1-{len(page_entries)} to view detail, {', '.join(nav_options)}[/dim]")
                
                # Get user input
                self.console.print("\n[cyan bold]BROWSE>[/cyan bold] ", end="")
                choice = self.safe_input()
                if choice is None:
                    break
                choice = choice.lower()
                
                if choice in ['q', 'quit']:
                    break
                elif choice == 'p' and current_page > 0:
                    current_page -= 1
                    continue
                elif choice == 'n' and current_page < total_pages - 1:
                    current_page += 1
                    continue
                else:
                    # Try to parse as entry number
                    try:
                        entry_num = int(choice)
                        if 1 <= entry_num <= len(page_entries):
                            detail_index = start_idx + (entry_num - 1)
                            mode = "detail"
                            self.console.clear()
                            continue
                        else:
                            self.console.print(f"[red]Invalid selection. Choose 1-{len(page_entries)}[/red]")
                    except ValueError:
                        self.console.print("[red]Invalid input. Enter a number, 'p', 'n', or 'q'.[/red]")
            
            elif mode == "detail":
                # Show help if toggled on
                if self.help_visible:
                    detail_help = """
[bold]Entry Detail Navigation:[/bold]

[green]Navigation Commands:[/green]
• [yellow]p[/yellow] - Previous entry
• [yellow]n[/yellow] - Next entry
• [yellow]l[/yellow] - Back to list view
• [yellow]q[/yellow] - Return to main menu

[green]Action Commands:[/green]
• [green]e[/green] - Edit this entry (refine with AI)
• [red]d[/red] - Delete this entry (soft delete - can be restored)

[green]Entry Sections:[/green]
• [cyan]Summary[/cyan]: AI-generated professional summary
• [cyan]Detailed Work[/cyan]: Specific tasks and activities performed
• [cyan]Impact Assessment[/cyan]: Business value and quantifiable metrics
• [cyan]Additional Info[/cyan]: Projects, collaborators, and tags
• [cyan]Original Input[/cyan]: Your raw notes (always preserved)

[green]Tips:[/green]
• Use edit (e) to improve entries with AI assistance
• Original input is never lost - always available for reference
• Impact ratings help prioritize accomplishments for reviews
                    """
                    self.console.print(Panel(detail_help.strip(), title="Detail View Help", border_style="blue"))
                    self.console.print()
                
                # Show detail view
                entry = entries[detail_index]
                processed = entry.processed
                
                # Show detailed entry
                self.console.print(f"\n[bold cyan]📖 ENTRY DETAIL - {detail_index + 1}/{len(entries)}[/bold cyan]")
                
                # Header with date and basic info
                date_display = DateParser.format_date_display(entry.date)
                impact_icon = {"individual": "👤", "team": "👥", "organization": "🏢"}.get(processed.impact.scope, "📝")
                
                header_info = f"{impact_icon} {date_display} | Impact: {processed.impact.scope.title()} ({processed.impact.significance}/5)"
                if entry.metadata.get("refinement_count", 0) > 0:
                    header_info += f" | Refined {entry.metadata['refinement_count']}x"
                
                self.console.print(f"[dim]{header_info}[/dim]")
                self.console.print(f"[dim]ID: {str(entry.id)[:8]}... | Created: {entry.timestamp.strftime('%Y-%m-%d %H:%M')}[/dim]\n")
                
                # Main content panels
                summary_panel = Panel(
                    processed.summary,
                    title="📝 Summary",
                    border_style="cyan"
                )
                self.console.print(summary_panel)
                
                work_panel = Panel(
                    processed.work,
                    title="🔧 Detailed Work",
                    border_style="blue"
                )
                self.console.print(work_panel)
                
                # Impact details
                impact_details = f"[bold]Business Value:[/bold] {processed.impact.business_value}"
                if processed.impact.metrics:
                    impact_details += f"\n[bold]Metrics:[/bold]\n" + "\n".join(f"• {metric}" for metric in processed.impact.metrics)
                
                impact_panel = Panel(
                    impact_details,
                    title="📊 Impact Assessment",
                    border_style="green"
                )
                self.console.print(impact_panel)
                
                # Additional info if present
                additional_info = []
                
                if processed.projects:
                    additional_info.append(f"[bold]📁 Projects:[/bold] {', '.join(processed.projects)}")
                
                if processed.collaborators:
                    additional_info.append(f"[bold]👥 Collaborators:[/bold] {', '.join(processed.collaborators)}")
                
                if processed.technical_details:
                    tech_details = "\n".join(f"• {detail}" for detail in processed.technical_details)
                    additional_info.append(f"[bold]🔧 Technical Details:[/bold]\n{tech_details}")
                
                if processed.tags:
                    additional_info.append(f"[bold]🏷️  Tags:[/bold] {', '.join(processed.tags)}")
                
                if additional_info:
                    additional_panel = Panel(
                        "\n\n".join(additional_info),
                        title="📋 Additional Information",
                        border_style="yellow"
                    )
                    self.console.print(additional_panel)
                
                # Raw input panel
                raw_panel = Panel(
                    entry.raw_input,
                    title="📄 Original Input",
                    border_style="dim"
                )
                self.console.print(raw_panel)
                
                # Navigation options
                nav_options = []
                if detail_index > 0:
                    nav_options.append("[yellow]p[/yellow]=previous")
                if detail_index < len(entries) - 1:
                    nav_options.append("[yellow]n[/yellow]=next")
                nav_options.extend(["[yellow]l[/yellow]=back to list", "[green]e[/green]=edit", "[red]d[/red]=delete", "[yellow]q[/yellow]=quit"])
                
                self.console.print(f"\n[dim]Navigation: {', '.join(nav_options)}[/dim]")
                
                # Get user input
                self.console.print("\n[cyan bold]DETAIL>[/cyan bold] ", end="")
                choice = self.safe_input()
                if choice is None:
                    break
                choice = choice.lower()
                
                if choice in ['q', 'quit']:
                    break
                elif choice == 'p' and detail_index > 0:
                    detail_index -= 1
                    self.console.clear()
                elif choice == 'n' and detail_index < len(entries) - 1:
                    detail_index += 1
                    self.console.clear()
                elif choice in ['l', 'list', 'back']:
                    mode = "list"
                    # Adjust current page to show the entry we were viewing
                    current_page = detail_index // page_size
                    self.console.clear()
                elif choice in ['e', 'edit']:
                    # Edit the current entry
                    edited_entry = self.edit_entry(entry)
                    if edited_entry:
                        # Update the entry in storage and refresh list
                        if self.storage.update_entry(edited_entry):
                            self.console.print("[green]✅ Entry updated successfully![/green]")
                            # Refresh entries list
                            entries = self.storage.load_all_entries(50)
                            # Find the updated entry's new position
                            for i, e in enumerate(entries):
                                if e.id == edited_entry.id:
                                    detail_index = i
                                    break
                            current_page = detail_index // page_size
                            self.console.clear()
                        else:
                            self.console.print("[red]❌ Failed to update entry.[/red]")
                elif choice in ['d', 'delete']:
                    # Confirm deletion
                    self.console.print(f"\n[bold red]⚠️  CONFIRM DELETE[/bold red]")
                    self.console.print(f"Are you sure you want to delete this entry?")
                    self.console.print(f"Date: {date_display}")
                    self.console.print(f"Summary: {processed.summary}")
                    self.console.print(f"[dim]Note: This is a soft delete - entry can be restored from Data Management → Backups[/dim]")
                    
                    self.console.print("\n[red bold]Confirm deletion? (y/n)>[/red bold] ", end="")
                    confirmation = self.safe_input()
                    if confirmation is None:
                        self.console.print("\n[yellow]Deletion cancelled.[/yellow]")
                        continue
                    confirmation = confirmation.lower()
                    
                    if confirmation in ['y']:
                        # Soft delete the entry
                        if self.storage.tombstone_entry(str(entry.id)):
                            self.console.print("[green]✅ Entry deleted successfully![/green]")
                            
                            # Refresh entries list (excluding tombstoned ones)
                            entries = self.storage.load_all_entries(50)
                            if not entries:
                                self.console.print("[dim]No more entries to view.[/dim]")
                                break
                            
                            # Adjust current index if needed
                            if detail_index >= len(entries):
                                detail_index = max(0, len(entries) - 1)
                            
                            # Refresh current page for list mode
                            current_page = detail_index // page_size
                            self.console.clear()
                        else:
                            self.console.print("[red]❌ Failed to delete entry.[/red]")
                    else:
                        self.console.print("[dim]Delete cancelled.[/dim]")
                else:
                    if choice == 'p' and detail_index == 0:
                        self.console.print("[yellow]Already at first entry.[/yellow]")
                    elif choice == 'n' and detail_index == len(entries) - 1:
                        self.console.print("[yellow]Already at last entry.[/yellow]")
                    else:
                        self.console.print("[red]Invalid option. Use 'p', 'n', 'l', 'e', 'd', or 'q'.[/red]")
    
    def edit_entry(self, entry: JournalEntry) -> Optional[JournalEntry]:
        """Edit an entry interactively. Returns updated entry or None if cancelled."""
        from datetime import datetime
        
        self.console.print(f"\n[bold green]✏️  EDIT ENTRY[/bold green]")
        date_display = DateParser.format_date_display(entry.date)
        self.console.print(f"[dim]Editing entry from {date_display}[/dim]\n")
        
        # Show current raw input
        self.console.print("[yellow]Current raw input:[/yellow]")
        raw_panel = Panel(
            entry.raw_input,
            title="Original Content",
            border_style="yellow"
        )
        self.console.print(raw_panel)
        
        self.console.print("\n[bold]Choose what to edit:[/bold]")
        self.console.print("  [cyan]1.[/cyan] Edit the raw text (inline)")
        self.console.print("  [cyan]2.[/cyan] Edit the raw text in editor ($EDITOR)")
        self.console.print("  [cyan]3.[/cyan] Just refine the current processed version")
        self.console.print("  [cyan]4.[/cyan] Change the date")
        self.console.print("  [cyan]q.[/cyan] Cancel editing")
        
        self.console.print("\n[green]Edit>[/green] ", end="")
        choice = self.safe_input()
        if choice is None:
            self.console.print("\n[yellow]Edit cancelled.[/yellow]")
            return None
        choice = choice.lower()
        
        if choice == 'q':
            return None
        elif choice == '1':
            # Edit raw text inline and re-process
            return self.edit_raw_text_inline(entry)
        elif choice == '2':
            # Edit raw text in editor and re-process
            return self.edit_raw_text_editor(entry)
        elif choice == '3':
            # Just refine current version
            return self.refine_existing_entry(entry)
        elif choice == '4':
            # Change date
            return self.change_entry_date(entry)
        else:
            self.console.print("[red]Invalid choice. Edit cancelled.[/red]")
            return None
    
    def edit_raw_text_inline(self, entry: JournalEntry) -> Optional[JournalEntry]:
        """Edit the raw text and re-process with LLM."""
        self.console.print(f"\n[bold]✏️  Edit Raw Text[/bold]")
        self.console.print("[dim]Enter your updated accomplishments (press Ctrl+D when done, Ctrl+C to cancel):[/dim]")
        self.console.print("[dim]Current text will be shown below for reference:[/dim]\n")
        
        # Show current text for reference
        current_panel = Panel(
            entry.raw_input,
            title="Current Text (for reference)",
            border_style="dim"
        )
        self.console.print(current_panel)
        
        self.console.print("\n[yellow]Enter new text:[/yellow]")
        
        # Collect new multi-line input
        lines = []
        try:
            while True:
                try:
                    line = input()
                    if line.strip().lower() == 'cancel':
                        self.console.print("[yellow]Edit cancelled.[/yellow]")
                        return None
                    lines.append(line)
                except KeyboardInterrupt:
                    # Ctrl+C pressed - cancel edit
                    self.console.print("\n[yellow]Edit cancelled.[/yellow]")
                    return None
                except EOFError:
                    # Ctrl+D pressed - end of input (finish)
                    if lines and any(line.strip() for line in lines):
                        # We have content, proceed with processing
                        break
                    else:
                        # No content, treat as cancel
                        self.console.print("\n[yellow]Edit cancelled (no content entered).[/yellow]")
                        return None
        except KeyboardInterrupt:
            # Ctrl+C pressed - always cancel
            self.console.print("\n[yellow]Edit cancelled.[/yellow]")
            return None
        
        new_raw_text = '\n'.join(lines).strip()
        
        if not new_raw_text:
            self.console.print("[yellow]No content entered. Edit cancelled.[/yellow]")
            return None
        
        # Re-process with LLM
        self.console.print("\n[cyan]🤖 Re-processing with LLM...[/cyan]")
        
        try:
            processor = EntryProcessor(storage=self.storage)
            
            # Show what we're sending for re-processing
            system_prompt = """You are a professional career development assistant. Your job is to help structure work accomplishments for promotion reviews.

Given a raw description of work, extract and structure the following information:

CRITICAL: For projects, be very selective. Only include work that contributes to a specific project with concrete deliverables."""
            
            context = processor._get_recent_context()
            context_section = f"\n\nFor context, here are recent projects and collaborators:\n{context}" if context else ""
            user_prompt = f"Extract and structure this work accomplishment:\n\n{new_raw_text}{context_section}"
            
            self.show_llm_query("Re-processing", system_prompt, user_prompt)
            
            model_info = self._get_current_model_info("processing")
            self.console.print(f"\n[dim]🤖⏳ Re-analyzing with {model_info}...[/dim]")
            
            # Create new entry with updated raw text
            updated_entry = JournalEntry(
                id=entry.id,  # Keep same ID
                timestamp=entry.timestamp,  # Keep original timestamp
                date=entry.date,  # Keep same date
                raw_input=new_raw_text,
                processed=processor._extract_structure(new_raw_text),
                jira_tickets=entry.jira_tickets,
                metadata={
                    **entry.metadata,
                    "last_edited": datetime.now().isoformat(),
                    "edit_count": entry.metadata.get("edit_count", 0) + 1
                },
                tombstoned=entry.tombstoned,
                tombstoned_at=entry.tombstoned_at
            )
            
            # Interactive refinement
            return self.interactive_refinement(updated_entry)
            
        except Exception as e:
            self.console.print(f"[red]❌ Error re-processing entry: {e}[/red]")
            return None
    
    def edit_raw_text_editor(self, entry: JournalEntry) -> Optional[JournalEntry]:
        """Edit the raw text in an external editor and re-process with LLM."""
        self.console.print(f"\n[bold]✏️  Edit Raw Text in Editor[/bold]")
        
        # Open editor with current text
        new_raw_text = self.open_editor(entry.raw_input)
        if new_raw_text is None:
            self.console.print("[yellow]Edit cancelled.[/yellow]")
            return None
        
        if not new_raw_text:
            self.console.print("[yellow]No content entered. Edit cancelled.[/yellow]")
            return None
        
        # Re-process with LLM
        self.console.print("\n[cyan]🤖 Re-processing with LLM...[/cyan]")
        
        try:
            processor = EntryProcessor(storage=self.storage)
            
            # Show what we're sending for re-processing
            system_prompt = """You are a professional career development assistant. Your job is to help structure work accomplishments for promotion reviews.

Given a raw description of work, extract and structure the following information:

CRITICAL: For projects, be very selective. Only include work that contributes to a specific project with concrete deliverables."""
            
            context = processor._get_recent_context()
            context_section = f"\n\nFor context, here are recent projects and collaborators:\n{context}" if context else ""
            user_prompt = f"Extract and structure this work accomplishment:\n\n{new_raw_text}{context_section}"
            
            self.show_llm_query("Re-processing", system_prompt, user_prompt)
            
            model_info = self._get_current_model_info("processing")
            self.console.print(f"\n[dim]🤖⏳ Re-analyzing with {model_info}...[/dim]")
            
            # Create new entry with updated raw text
            updated_entry = JournalEntry(
                id=entry.id,  # Keep same ID
                timestamp=entry.timestamp,  # Keep original timestamp
                date=entry.date,  # Keep same date
                raw_input=new_raw_text,
                processed=processor._extract_structure(new_raw_text),
                jira_tickets=entry.jira_tickets,
                metadata={
                    **entry.metadata,
                    "last_edited": datetime.now().isoformat(),
                    "edit_count": entry.metadata.get("edit_count", 0) + 1
                },
                tombstoned=entry.tombstoned,
                tombstoned_at=entry.tombstoned_at
            )
            
            # Interactive refinement
            return self.interactive_refinement(updated_entry)
            
        except Exception as e:
            self.console.print(f"[red]❌ Error re-processing entry: {e}[/red]")
            return None
    
    def refine_existing_entry(self, entry: JournalEntry) -> Optional[JournalEntry]:
        """Refine the existing processed entry."""
        self.console.print(f"\n[bold]🔧 Refine Entry[/bold]")
        self.console.print("[yellow]What would you like to improve or change?[/yellow]")
        self.console.print("[dim]Examples: 'Make the summary more impactful', 'Add technical details about Docker', 'Include Sarah as collaborator'[/dim]")
        self.console.print("[yellow]Refinement>[/yellow] ", end="")
        refinement = self.safe_input()
        if refinement is None:
            self.console.print("\n[yellow]Refinement cancelled.[/yellow]")
            return None
        
        if not refinement:
            self.console.print("[yellow]No refinement provided. Edit cancelled.[/yellow]")
            return None
        
        try:
            processor = EntryProcessor(storage=self.storage)
            
            # Show refinement query to user
            system_prompt = """You are helping refine a journal entry based on user feedback. 
Your job is to improve the existing structured entry according to the user's instructions.

Keep the same JSON structure but modify the content based on the feedback.
Be precise and maintain professional tone suitable for promotion reviews."""
            
            current_processed = entry.processed
            import json  # Ensure json is available in local scope
            user_prompt = f"""Current entry:
{json.dumps(current_processed.model_dump(), indent=2)}

Original raw input:
{entry.raw_input}

User feedback: {refinement}

Please refine the entry based on this feedback. Respond ONLY with valid JSON in the same format."""
            
            self.show_llm_query("Entry Refinement", system_prompt, user_prompt)
            
            with self.llm_spinner("🤖 Refining entry", "processing"):
                updated_entry = processor.refine_entry(entry, refinement)
            
            # Update metadata
            updated_entry.metadata["last_edited"] = datetime.now().isoformat()
            updated_entry.metadata["edit_count"] = entry.metadata.get("edit_count", 0) + 1
            
            return updated_entry
            
        except Exception as e:
            self.console.print(f"[red]❌ Error refining entry: {e}[/red]")
            return None
    
    def change_entry_date(self, entry: JournalEntry) -> Optional[JournalEntry]:
        """Change the date of an entry."""
        self.console.print(f"\n[bold]📅 Change Date[/bold]")
        current_date = DateParser.format_date_display(entry.date)
        self.console.print(f"Current date: {current_date}")
        
        # Date input with retry loop
        new_date = None
        while new_date is None:
            self.console.print("[dim]Enter new date (e.g., 'today', 'yesterday', 'last monday', '2025-08-28'):[/dim]")
            self.console.print("[green]New Date>[/green] ", end="")
            date_input = self.safe_input()
            if date_input is None:
                self.console.print("\n[yellow]Date change cancelled.[/yellow]")
                return None
            
            if not date_input:
                self.console.print("[yellow]No date provided. Please try again.[/yellow]\n")
                continue
            
            # Try to parse date
            try:
                new_date = self.parse_date_with_llm(date_input)
                new_date_display = DateParser.format_date_display(new_date)
            except ValueError as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
                self.console.print("[yellow]Please try again.[/yellow]\n")
        
        # Once we have a valid date, confirm the change
        self.console.print(f"[green]Date will be changed to: {new_date_display}[/green]")
        self.console.print("[yellow]Confirm? (y/n)>[/yellow] ", end="")
        confirm = self.safe_input()
        if confirm is None:
            self.console.print("\n[yellow]Date change cancelled.[/yellow]")
            return None
        confirm = confirm.lower()
        
        if confirm in ['y']:
            # Create updated entry with new date
            from datetime import datetime
            updated_entry = JournalEntry(
                id=entry.id,
                timestamp=entry.timestamp,
                date=new_date,  # Updated date
                raw_input=entry.raw_input,
                processed=entry.processed,
                jira_tickets=entry.jira_tickets,
                metadata={
                    **entry.metadata,
                    "last_edited": datetime.now().isoformat(),
                    "edit_count": entry.metadata.get("edit_count", 0) + 1,
                    "date_changed": True
                },
                tombstoned=entry.tombstoned,
                tombstoned_at=entry.tombstoned_at
            )
            return updated_entry
        else:
            self.console.print("[yellow]Date change cancelled.[/yellow]")
            return None
    
    def interactive_refinement(self, entry: JournalEntry) -> Optional[JournalEntry]:
        """Interactive refinement loop for edited entries."""
        while True:
            # Show current processed entry
            processed = entry.processed
            summary_panel = Panel(
                f"[bold]{processed.summary}[/bold]\n\n"
                f"Impact: {processed.impact.scope.title()} ({processed.impact.significance}/5)\n"
                f"Business Value: {processed.impact.business_value}\n"
                f"Projects: {', '.join(processed.projects) if processed.projects else 'None specified'}\n"
                f"Technical Details: {', '.join(processed.technical_details) if processed.technical_details else 'None specified'}\n"
                f"Collaborators: {', '.join(processed.collaborators) if processed.collaborators else 'Solo work'}\n"
                f"Tags: {', '.join(processed.tags) if processed.tags else 'None'}",
                title="Updated Entry Preview",
                border_style="green"
            )
            self.console.print(summary_panel)
            
            # Ask for feedback
            self.console.print("\n[dim]Options: [green]s[/green]=save, [yellow]r[/yellow]=refine, [red]c[/red]=cancel[/dim]")
            self.console.print("[green]Action>[/green] ", end="")
            action = self.safe_input()
            if action is None:
                self.console.print("\n[yellow]Edit cancelled.[/yellow]")
                return None
            action = action.lower()
            
            if action in ['s', '']:
                return entry
                
            elif action in ['r']:
                # Get refinement instructions
                self.console.print("\n[yellow]What would you like to improve or change?[/yellow]")
                self.console.print("[yellow]Refinement>[/yellow] ", end="")
                refinement = self.safe_input()
                if refinement is None:
                    self.console.print("\n[yellow]Refinement cancelled.[/yellow]")
                    continue
                
                if refinement:
                    # Show refinement query to user
                    system_prompt = """You are helping refine a journal entry based on user feedback. 
Your job is to improve the existing structured entry according to the user's instructions.

Keep the same JSON structure but modify the content based on the feedback.
Be precise and maintain professional tone suitable for promotion reviews."""
                    
                    current_processed = entry.processed
                    import json  # Ensure json is available in local scope
                    user_prompt = f"""Current entry:
{json.dumps(current_processed.model_dump(), indent=2)}

Original raw input:
{entry.raw_input}

User feedback: {refinement}

Please refine the entry based on this feedback. Respond ONLY with valid JSON in the same format."""
                    
                    self.show_llm_query("Entry Refinement", system_prompt, user_prompt)
                    
                    # Refine with LLM
                    model_info = self._get_current_model_info("processing")
                    try:
                        processor = EntryProcessor(storage=self.storage)
                        with self.llm_spinner("🤖 Refining entry", "processing"):
                            entry = processor.refine_entry(entry, refinement)
                    except Exception as e:
                        self.console.print(f"[red]❌ Error refining: {e}[/red]")
                
            elif action in ['c']:
                self.console.print("[yellow]Edit cancelled.[/yellow]")
                return None
            else:
                self.console.print("[red]Invalid option. Use 's', 'r', or 'c'.[/red]")
    
    def generate_summary(self):
        """Generate a 1:1 summary for a specified date range."""
        self.console.clear()
        self.console.print("\n[bold cyan]📊 Generate 1:1 Summary[/bold cyan]")
        self.console.print("Create a summary of your accomplishments for your manager.")
        self.console.print("[dim]Includes Top 3 highlights with emojis for quick discussion points.[/dim]")
        
        # Show help if toggled on
        if self.help_visible:
            help_text = """
[bold]1:1 Summary Generation Help:[/bold]

[green]What this creates:[/green]
• Professional summary of your work accomplishments
• Top 3 highlights with emoji indicators for easy discussion  
• Business impact focus suitable for manager conversations
• Structured format covering achievements, collaboration, and growth

[green]Time Period Options:[/green]
• [cyan]Last Week[/cyan]: Perfect for weekly 1:1s and standup updates
• [cyan]Last 2 Weeks[/cyan]: Good for bi-weekly check-ins and sprint reviews
• [cyan]Last Month[/cyan]: Ideal for monthly reviews and milestone summaries
• [cyan]Custom Range[/cyan]: Choose specific dates (coming soon)

[green]Pro Tips:[/green]
• Summaries work best with 3+ entries in the selected timeframe
• After generation, you can refine the content with specific feedback
• Generated summaries can be saved as text files for your records
• Focus is on career growth and business value delivery
            """
            self.console.print(Panel(help_text.strip(), title="1:1 Summary Help", border_style="blue"))
            self.console.print()
        
        # Date range selection
        self.console.print("\n[bold]Select time period:[/bold]")
        self.console.print("  [cyan]1.[/cyan] Last week (7 days)")
        self.console.print("  [cyan]2.[/cyan] Last 2 weeks (14 days)")
        self.console.print("  [cyan]3.[/cyan] Last month (30 days)")
        self.console.print("  [cyan]4.[/cyan] Custom date range")
        self.console.print("  [cyan]h.[/cyan] Toggle help")
        self.console.print("  [cyan]q.[/cyan] Cancel")
        
        while True:
            self.console.print("\n[green]Time period>[/green] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                self.console.print("\n[yellow]Summary cancelled.[/yellow]")
                return
            
            choice = choice.lower()
            
            # Handle help toggle
            if choice == 'h':
                self.help_visible = not self.help_visible
                self.console.clear()
                self.console.print("\n[bold cyan]📊 Generate 1:1 Summary[/bold cyan]")
                self.console.print("Create a summary of your accomplishments for your manager.")
                self.console.print("[dim]Includes Top 3 highlights with emojis for quick discussion points.[/dim]")
                
                # Show help if toggled on
                if self.help_visible:
                    help_text = """
[bold]1:1 Summary Generation Help:[/bold]

[green]What this creates:[/green]
• Professional summary of your work accomplishments
• Top 3 highlights with emoji indicators for easy discussion  
• Business impact focus suitable for manager conversations
• Structured format covering achievements, collaboration, and growth

[green]Time Period Options:[/green]
• [cyan]Last Week[/cyan]: Perfect for weekly 1:1s and standup updates
• [cyan]Last 2 Weeks[/cyan]: Good for bi-weekly check-ins and sprint reviews
• [cyan]Last Month[/cyan]: Ideal for monthly reviews and milestone summaries
• [cyan]Custom Range[/cyan]: Choose specific dates (coming soon)

[green]Pro Tips:[/green]
• Summaries work best with 3+ entries in the selected timeframe
• After generation, you can refine the content with specific feedback
• Generated summaries can be saved as text files for your records
• Focus is on career growth and business value delivery
                    """
                    self.console.print(Panel(help_text.strip(), title="1:1 Summary Help", border_style="blue"))
                    self.console.print()
                
                # Redisplay menu
                self.console.print("\n[bold]Select time period:[/bold]")
                self.console.print("  [cyan]1.[/cyan] Last week (7 days)")
                self.console.print("  [cyan]2.[/cyan] Last 2 weeks (14 days)")
                self.console.print("  [cyan]3.[/cyan] Last month (30 days)")
                self.console.print("  [cyan]4.[/cyan] Custom date range")
                self.console.print("  [cyan]h.[/cyan] Toggle help")
                self.console.print("  [cyan]q.[/cyan] Cancel")
                continue
            
            # Calculate date range
            from datetime import datetime, timedelta
            today = datetime.now()
            
            if choice == '1':
                days_back = 7
                period_name = "Last Week"
                break
            elif choice == '2':
                days_back = 14
                period_name = "Last 2 Weeks"
                break
            elif choice == '3':
                days_back = 30
                period_name = "Last Month"
                break
            elif choice == '4':
                # Custom date range - for now, default to 14 days, we can enhance this later
                days_back = 14
                period_name = "Custom Period"
                self.console.print("[yellow]Custom date ranges coming soon! Using last 2 weeks for now.[/yellow]")
                break
            else:
                self.console.print("[red]Invalid choice. Try 1-4, h for help, or q to cancel.[/red]")
                continue
        
        start_date = today - timedelta(days=days_back)
        
        # Load entries from the date range
        entries = self._load_entries_in_range(start_date, today)
        
        if not entries:
            self.console.print(f"[yellow]No entries found for {period_name.lower()}. Add some entries first![/yellow]")
            return
        
        self.console.print(f"\n[dim]Found {len(entries)} entries for {period_name.lower()}[/dim]")
        
        # Generate summary with LLM
        summary = self._generate_llm_summary(entries, period_name, start_date, today)
        if not summary:
            self.console.print("[red]❌ Failed to generate summary.[/red]")
            return
        
        # Display summary
        self._display_summary(summary, period_name)
        
        # Options to save or refine
        while True:
            # Show action help if toggled
            if self.help_visible:
                action_help = """
[bold]Summary Actions Help:[/bold]

[green]Available Actions:[/green]
• [green]Save (s)[/green]: Save summary to text file in your work journal directory
• [yellow]Refine (r)[/yellow]: Improve the summary with AI using your specific feedback
• [red]Cancel (c)[/red]: Discard the summary and return to main menu
• [cyan]Help (h)[/cyan]: Toggle this help display

[green]Refinement Examples:[/green]
• "Make it more concise" - Shorter, punchier format
• "Add more technical details" - Include specific technologies and approaches
• "Focus on business impact" - Emphasize ROI and business value
• "Make it sound more confident" - Stronger, more assertive language
• "Add metrics where possible" - Include quantifiable achievements

[green]Tips:[/green]
• You can refine multiple times until you're satisfied with the result
• Saved summaries include timestamp and can be found in your journal directory
• Use specific feedback for better refinement results
                """
                self.console.print(Panel(action_help.strip(), title="Summary Actions Help", border_style="blue"))
                self.console.print()
            
            self.console.print("\n[dim]Options: [green]s[/green]=save, [yellow]r[/yellow]=refine, [cyan]h[/cyan]=help, [red]c[/red]=cancel[/dim]")
            self.console.print("[cyan]Action>[/cyan] ", end="")
            action = self.safe_input()
            if action is None:
                break
            action = action.lower()
            
            if action == 'h':
                self.help_visible = not self.help_visible
                self.console.clear()
                self._display_summary(summary, period_name)
                continue
            elif action in ['s', '']:
                self._save_summary(summary, period_name)
                break
            elif action in ['r']:
                # Get refinement instructions
                self.console.print("\n[yellow]How would you like to improve the summary?[/yellow]")
                self.console.print("[dim]Examples: 'Make it more concise', 'Add more technical details', 'Focus on business impact'[/dim]")
                self.console.print("[yellow]Refinement>[/yellow] ", end="")
                refinement = self.safe_input()
                if refinement:
                    refined_summary = self._refine_summary(summary, refinement, entries)
                    if refined_summary:
                        summary = refined_summary
                        self._display_summary(summary, period_name)
            elif action in ['c']:
                self.console.print("[yellow]Summary cancelled.[/yellow]")
                break
            else:
                self.console.print("[red]Invalid option. Use 's', 'r', 'h', or 'c'.[/red]")
    
    def _load_entries_in_range(self, start_date, end_date):
        """Load entries within a date range."""
        from datetime import datetime
        
        # Load all entries (we could optimize this later to only load relevant files)
        all_entries = self.storage.load_all_entries(100)  # Increased limit for summaries
        
        # Filter by date range
        filtered_entries = []
        for entry in all_entries:
            entry_date = datetime.strptime(entry.date, '%Y-%m-%d')
            if start_date.date() <= entry_date.date() <= end_date.date():
                filtered_entries.append(entry)
        
        # Sort by date (oldest first for chronological summary)
        filtered_entries.sort(key=lambda e: e.date)
        return filtered_entries
    
    def _generate_llm_summary(self, entries, period_name, start_date, end_date):
        """Generate a manager-friendly summary using LLM."""
        from .entry_processor import EntryProcessor
        
        # Prepare data for LLM
        entries_data = []
        for entry in entries:
            processed = entry.processed
            entry_data = {
                "date": entry.date,
                "summary": processed.summary,
                "work": processed.work,
                "projects": processed.projects,
                "collaborators": processed.collaborators,
                "impact": {
                    "scope": processed.impact.scope,
                    "significance": processed.impact.significance,
                    "business_value": processed.impact.business_value,
                    "metrics": processed.impact.metrics
                },
                "technical_details": processed.technical_details,
                "tags": processed.tags
            }
            entries_data.append(entry_data)
        
        # Create summary prompt
        system_prompt = """You are a professional career development assistant helping create executive summaries for 1:1 meetings with managers.

Create a concise, manager-friendly summary that highlights:
1. Key accomplishments and their business impact
2. Projects worked on and progress made  
3. Collaboration and team contributions
4. Technical achievements and innovations
5. Metrics and quantifiable results

CRITICAL: Start the summary with a "🎯 Top 3 Highlights" section that lists the 3 most impactful or important accomplishments from this period. Each item should be a single, punchy sentence with an appropriate emoji. These are the key talking points for a time-constrained 1:1 meeting.

Format the summary in markdown with clear sections:
- 🎯 Top 3 Highlights (3 bullet points with emojis)
- Key Accomplishments 
- Project Progress
- Collaboration & Impact
- Looking Ahead (if applicable)

Focus on business value and impact rather than technical details. Keep it concise but comprehensive - aim for 1-2 pages that a manager can quickly scan.

Use a professional, confident tone suitable for career discussions."""

        entries_json = json.dumps(entries_data, indent=2)
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        user_prompt = f"""Generate a 1:1 summary for {period_name} ({date_range}) based on these work entries:

{entries_json}

Create a professional summary that highlights key accomplishments, business impact, and growth. 

IMPORTANT: Start with "🎯 Top 3 Highlights" - identify the 3 most impactful accomplishments that should definitely be discussed in a time-constrained 1:1 meeting. Make each highlight punchy and include relevant emojis.

Structure it with clear sections and focus on what would be most relevant for a manager to understand my contributions and progress."""

        # Show what we're sending to LLM
        self.show_llm_query("1:1 Summary Generation", system_prompt, user_prompt)
        
        try:
            processor = EntryProcessor(storage=self.storage)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            with self.llm_spinner("🤖 Generating summary", "processing"):
                response = processor.llm_client.call_llm("processing", messages, max_tokens=2000, temperature=0.3)
            return response.strip()
            
        except Exception as e:
            self.console.print(f"[red]❌ Error generating summary: {e}[/red]")
            return None
    
    def _display_summary(self, summary, period_name):
        """Display the generated summary."""
        summary_panel = Panel(
            summary,
            title=f"📊 1:1 Summary - {period_name}",
            border_style="green",
            expand=True
        )
        self.console.print(summary_panel)
    
    def _save_summary(self, summary, period_name):
        """Save summary to a file."""
        from datetime import datetime
        from pathlib import Path
        
        # Create exports directory if it doesn't exist  
        exports_dir = self.storage.exports_path
        exports_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"1on1_summary_{period_name.lower().replace(' ', '_')}_{timestamp}.md"
        filepath = exports_dir / filename
        
        # Add header to summary
        header = f"""# 1:1 Summary - {period_name}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        
        try:
            with open(filepath, 'w') as f:
                f.write(header + summary)
            
            self.console.print(f"[green]✅ Summary saved to: {filepath}[/green]")
            
            # Try to open the file in the default app
            import subprocess
            import sys
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", str(filepath)], check=True)
                elif sys.platform == "linux":
                    subprocess.run(["xdg-open", str(filepath)], check=True)
                elif sys.platform == "win32":
                    subprocess.run(["start", str(filepath)], shell=True, check=True)
                self.console.print("[dim]File opened in default application.[/dim]")
            except:
                self.console.print(f"[dim]You can find the file at: {filepath}[/dim]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Error saving summary: {e}[/red]")
    
    def _refine_summary(self, summary, refinement_instruction, entries):
        """Refine the summary based on user feedback."""
        from .entry_processor import EntryProcessor
        
        system_prompt = """You are helping refine a 1:1 summary based on user feedback.
        
Modify the summary according to the user's instructions while maintaining the professional, manager-friendly tone and markdown formatting. Keep the focus on business impact and career growth."""

        user_prompt = f"""Current summary:
{summary}

User feedback: {refinement_instruction}

Please refine the summary based on this feedback. Maintain the markdown structure and professional tone."""

        # Show refinement query
        self.show_llm_query("Summary Refinement", system_prompt, user_prompt)
        
        try:
            processor = EntryProcessor(storage=self.storage)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            with self.llm_spinner("🤖 Refining summary", "processing"):
                response = processor.llm_client.call_llm("processing", messages, max_tokens=2000, temperature=0.3)
            return response.strip()
            
        except Exception as e:
            self.console.print(f"[red]❌ Error refining summary: {e}[/red]")
            return None
    
    def delete_entry(self):
        """Delete an entry with paged selection."""
        self.console.clear()
        entries = self.storage.load_all_entries(50)
        
        if not entries:
            self.console.print("[yellow]No entries found to delete.[/yellow]")
            return
        
        page_size = 10
        current_page = 0
        total_pages = (len(entries) - 1) // page_size + 1
        
        while True:
            # Calculate page bounds
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(entries))
            page_entries = entries[start_idx:end_idx]
            
            # Show current page
            self.console.print(f"\n[bold red]🗑️  DELETE MODE - Page {current_page + 1}/{total_pages}[/bold red]")
            self.console.print("[dim]Select an entry to delete, or navigate pages:[/dim]\n")
            
            for i, entry in enumerate(page_entries, 1):
                processed = entry.processed
                impact_icon = {"individual": "👤", "team": "👥", "organization": "🏢"}.get(processed.impact.scope, "📝")
                date_display = DateParser.format_date_display(entry.date)
                
                # Show entry number (1-10), ID first 8 chars, and summary
                entry_id_short = str(entry.id)[:8]
                self.console.print(f"  [cyan]{i:2d}.[/cyan] {impact_icon} {date_display} [dim]({entry_id_short})[/dim]")
                self.console.print(f"      {processed.summary}")
                
                if processed.collaborators:
                    collab_list = ', '.join(processed.collaborators[:2])
                    if len(processed.collaborators) > 2:
                        collab_list += f" +{len(processed.collaborators)-2} more"
                    self.console.print(f"      [dim]👥 {collab_list}[/dim]")
                
                self.console.print()
            
            # Show navigation options
            nav_options = []
            if current_page > 0:
                nav_options.append("[yellow]p[/yellow]=prev page")
            if current_page < total_pages - 1:
                nav_options.append("[yellow]n[/yellow]=next page")
            nav_options.append("[yellow]q[/yellow]=quit")
            
            self.console.print(f"[dim]Options: 1-{len(page_entries)} to delete, {', '.join(nav_options)}[/dim]")
            
            # Get user input
            self.console.print("\n[red bold]⚠️  DELETE>[/red bold] ", end="")
            choice = self.safe_input()
            if choice is None:
                break
            choice = choice.lower()
            
            if choice == 'q':
                self.console.print("[dim]Delete cancelled.[/dim]")
                return
            elif choice == 'p' and current_page > 0:
                current_page -= 1
                continue
            elif choice == 'n' and current_page < total_pages - 1:
                current_page += 1
                continue
            else:
                # Try to parse as entry number
                try:
                    entry_num = int(choice)
                    if 1 <= entry_num <= len(page_entries):
                        selected_entry = page_entries[entry_num - 1]
                        if self.confirm_delete(selected_entry):
                            if self.storage.delete_entry(str(selected_entry.id)):
                                self.console.print("[green]✅ Entry deleted successfully![/green]")
                                # Refresh entries list
                                entries = self.storage.load_all_entries(50)
                                if not entries:
                                    self.console.print("[dim]No more entries to delete.[/dim]")
                                    return
                                
                                # Adjust current page if needed
                                total_pages = (len(entries) - 1) // page_size + 1
                                if current_page >= total_pages:
                                    current_page = max(0, total_pages - 1)
                            else:
                                self.console.print("[red]❌ Failed to delete entry.[/red]")
                        continue
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(page_entries)}[/red]")
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number, 'p', 'n', or 'q'.[/red]")
    
    def confirm_delete(self, entry: 'JournalEntry') -> bool:
        """Confirm deletion of an entry."""
        processed = entry.processed
        date_display = DateParser.format_date_display(entry.date)
        
        # Show entry details
        self.console.print(f"\n[bold red]⚠️  CONFIRM DELETE[/bold red]")
        self.console.print(f"Date: {date_display}")
        self.console.print(f"Summary: {processed.summary}")
        if processed.collaborators:
            self.console.print(f"Collaborators: {', '.join(processed.collaborators)}")
        self.console.print(f"[dim]Note: This is a soft delete - entry can be restored from Data Management → Backups[/dim]")
        
        # Get confirmation
        self.console.print("\n[red bold]Delete this entry? (y/n)>[/red bold] ", end="")
        confirmation = self.safe_input()
        if confirmation is None:
            return False
        
        return confirmation.lower() in ['y']
    
    def clear_screen(self):
        """Clear the screen and show welcome again."""
        self.console.clear()
        self.show_welcome()
    
    def show_help(self, context="main"):
        """Show contextual help information."""
        if context == "main":
            help_text = """
[bold]Daily Workflow:[/bold]
[green]a[/green] - Add accomplishments for any date
    Write what you did, AI extracts key details and impact

[green]l[/green] - Browse your work history  
    Paginated list with search and detailed entry views

[green]s[/green] - Create manager-ready summaries
    Generate Top 3 highlights for 1:1 meetings, exported as markdown (has help)

[green]x[/green] - Access settings and data management
    Model configuration, entity management, backups (has help)

[bold]Navigation:[/bold]
[green]h[/green] - Context-sensitive help  •  [green]c[/green] - Clear screen  •  [green]q[/green] - Quit

[bold]Entry Tips:[/bold]
• Write multiple detailed lines describing your work
• Use natural dates: 'yesterday', 'last friday', '2024-01-15'
• Ctrl+D to finish entry, Ctrl+C to cancel anytime
            """
        elif context == "system":
            help_text = """
[bold]Configuration:[/bold]
[green]m[/green] - Switch AI models and test connections
    Local (Ollama) vs cloud providers, preset configurations

[bold]Data Management:[/bold]
[green]e[/green] - Manage collaborators, projects, and tags
    Edit, merge, or discover entities from your entries

[green]d[/green] - Remove unwanted entries  
    Browse and soft delete with confirmation (can be restored from data management)

[green]b[/green] - Backup and restore your data
    Create snapshots, restore from backups, manage deleted entries (recycle bin)

[bold]Navigation:[/bold]
[green]h[/green] - Show this help  •  [green]c[/green] - Clear  •  [green]q[/green] - Back to main
            """
        elif context == "model_settings":
            help_text = """
[bold]Model Configuration Help:[/bold]

[yellow]Configurations[/yellow] - Switch between saved setups
• Choose 1-9 to switch between named configurations
• Each configuration assigns models to the three functions
• Current configuration shown at top with ← marker

[yellow]Create New Configuration (1)[/yellow] - Guided setup process
• Step 1: Add providers (OpenAI, Anthropic, Ollama, etc.)
• Step 2: Select models from each provider
• Step 3: Create configuration by assigning models to functions
• Step 4: Activate your new configuration

[yellow]Test Current Setup (t)[/yellow] - Verify active configuration
• Test if your current models are reachable
• Check for authentication issues  
• Validate configuration before use

[bold]Model Functions Explained:[/bold]
• [green]Conversation[/green]: Interactive chat and entry refinement
• [green]Processing[/green]: Structures raw entries into professional summaries
• [green]JIRA Matching[/green]: Finds relevant tickets for your work

[bold]Quick Tips:[/bold]
• Use Claude 3.5 Sonnet or GPT-4 for best processing quality
• GPT-3.5-turbo works well for JIRA matching (faster and cheaper)
• Local models (Ollama, LM Studio) work great for privacy-focused workflows
            """
        else:
            help_text = f"[yellow]Help not available for context: {context}[/yellow]"
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def system_menu(self):
        """Show system menu with all settings and advanced tools."""
        while True:
            # Clear screen for each menu display
            self.console.clear()
            
            self.console.print(f"\n[bold cyan]⚙️  System Menu[/bold cyan]")
            
            # Show help if toggled on
            if self.help_visible:
                help_text = """
[bold]System Menu Help:[/bold]

[green]Settings:[/green]
• [green]Model & Settings (m)[/green]: Configure LLM providers and model assignments
  - Add/test providers (OpenAI, Anthropic, Ollama, custom)
  - Create configurations for different workflows
  - Switch between saved configurations

[green]Data Management:[/green]
• [green]Manage Entities (e)[/green]: Clean up collaborators, projects, and tags
  - Edit names and aliases for consistency
  - Merge duplicate entries
  - Discover new entities from your journal entries

• [green]Delete Entries (d)[/green]: Remove unwanted journal entries
  - Browse entries with pagination
  - Soft delete with confirmation (can be restored)
  - Access through recycle bin in backups

• [green]Data Management (b)[/green]: Backup, restore, and maintenance
  - Create/restore backups of your entire journal
  - Manage deleted entries (recycle bin)
  - Bulk reprocessing and data integrity tools

[green]Navigation Tips:[/green]
• All changes are saved automatically
• Use 'h' to toggle this help on/off
• Most operations have their own contextual help
                """
                self.console.print(Panel(help_text.strip(), title="System Menu Help", border_style="blue"))
                self.console.print()
            
            # Show system options
            help_text = "Help (hide)" if self.help_visible else "Help (show)"
            system_text = f"""
[bold]Settings:[/bold]
  [green]m[/green] - Model & settings

[bold]Data Management:[/bold]
  [green]e[/green] - Manage entities (collaborators, projects, tags)
  [green]d[/green] - Delete entries (soft delete - can be restored)
  [green]b[/green] - Data management & backups

[bold]Navigation:[/bold]
  [green]h[/green] - {help_text}
  [green]c[/green] - Clear screen
  [green]q[/green] - Back to main menu
            """
            
            self.console.print(system_text)
            
            # Get user input
            self.console.print("\n[cyan bold]SYSTEM>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            choice = choice.lower()
            
            if choice == 'm':
                self.model_settings()
            elif choice == 'e':
                self.manage_entities()
            elif choice == 'd':
                self.delete_entry()
            elif choice == 'b':
                self.data_management()
            elif choice == 'h':
                self.help_visible = not self.help_visible
                # Menu will redraw with help toggled on next iteration
            elif choice == 'c':
                self.clear_screen()
            elif choice == '':
                # Empty command, just redraw menu
                pass
            else:
                self.console.print(f"[red]Unknown command: '{choice}'. Use single letters from the menu.[/red]")

    def model_settings(self):
        """Simple model settings menu."""
        from .llm import LLMClient
        
        while True:
            # Clear screen for each menu display
            self.console.clear()
            
            self.console.print(f"\n[bold cyan]🤖 Model Settings[/bold cyan]")
            
            # Get current settings
            llm_client = LLMClient()
            settings = llm_client.settings
            
            # Show current configuration
            if settings.current_config and settings.current_config in settings.configurations:
                current_config = settings.configurations[settings.current_config]
                settings_text = f"""[bold]Current Configuration: {settings.current_config}[/bold]
  💬 Conversation: {current_config.conversation.provider} / {current_config.conversation.model}
  ⚙️  Processing: {current_config.processing.provider} / {current_config.processing.model}
  🎫 JIRA Matching: {current_config.jira_matching.provider} / {current_config.jira_matching.model}"""
            else:
                settings_text = f"""[bold]Current Configuration: [red]None[/red][/bold]
  [red]No configuration is active![/red]
  [dim]You need to create a configuration.[/dim]"""
            
            self.console.print(Panel(settings_text.strip(), title="Active Configuration", border_style="cyan"))
            
            self.console.print("\nWhat would you like to do?")
            
            # Show simple menu based on our design
            menu_text = """
[bold]Model Settings:[/bold]
  [green]1.[/green] 🔄 Switch Configuration
  [green]2.[/green] ➕ Create New Configuration  
  [green]3.[/green] 🧪 Test Current Configuration
  [green]4.[/green] 📖 Help
            """
            
            self.console.print(menu_text)
            
            # Get user input
            self.console.print("\n[cyan bold]MODEL>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            
            if choice == '1':
                # Switch Configuration
                self._switch_configuration(llm_client)
            elif choice == '2':
                # Create New Configuration - Run our 4-step builder
                success = self._run_config_builder()
                if success:
                    self.console.print("\n[green]Configuration created successfully![/green]")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    self.safe_input()
            elif choice == '3':
                # Test Current Configuration
                self._test_current_configuration(llm_client)
            elif choice == '4' or choice.lower() == 'h':
                # Help
                self._show_simple_model_settings_help()
            else:
                self.console.print("[red]Invalid choice. Please select 1-4, or 'q' to go back.[/red]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
    
    def _switch_configuration(self, llm_client: 'LLMClient'):
        """Switch to a different configuration."""
        self.console.clear()
        self.console.print("\n[bold cyan]🔄 Switch Configuration[/bold cyan]")
        
        if not llm_client.settings.configurations:
            self.console.print("\n[yellow]No configurations available![/yellow]")
            self.console.print("You need to create a configuration first using option 2.")
            self.console.print("\n[dim]Press Enter to go back...[/dim]")
            self.safe_input()
            return
        
        configs = list(llm_client.settings.configurations.keys())
        current = llm_client.settings.current_config
        
        self.console.print(f"[dim]Current: {current or 'None'}[/dim]")
        
        self.console.print("\n[bold]Available Configurations:[/bold]")
        for i, config_name in enumerate(configs, 1):
            marker = " [green]← current[/green]" if config_name == current else ""
            self.console.print(f"  {i}. {config_name}{marker}")
        
        while True:
            choice = self.safe_input(f"\nChoose configuration (1-{len(configs)}, or 'q' to cancel): ").strip()
            
            if choice.lower() == 'q':
                return
                
            if choice.isdigit() and 1 <= int(choice) <= len(configs):
                selected_config = configs[int(choice) - 1]
                
                if selected_config == current:
                    self.console.print(f"[yellow]'{selected_config}' is already the current configuration.[/yellow]")
                else:
                    llm_client.settings.current_config = selected_config
                    llm_client.save_settings()
                    self.console.print(f"[green]✅ Switched to configuration: {selected_config}[/green]")
                
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
                return
            else:
                self.console.print(f"[red]Invalid choice. Please enter 1-{len(configs)}.[/red]")
    
    def _test_current_configuration(self, llm_client: 'LLMClient'):
        """Test the current configuration."""
        self.console.clear()
        self.console.print("\n[bold cyan]🧪 Test Current Configuration[/bold cyan]")
        
        if not llm_client.settings.current_config:
            self.console.print("\n[red]No configuration is currently active![/red]")
            self.console.print("Please create or switch to a configuration first.")
            self.console.print("\n[dim]Press Enter to go back...[/dim]")
            self.safe_input()
            return
        
        config_name = llm_client.settings.current_config
        config = llm_client.settings.configurations[config_name]
        
        self.console.print(f"Testing configuration: [bold]{config_name}[/bold]")
        
        # Test each need
        needs = [
            ("conversation", "💬 Conversation", config.conversation),
            ("processing", "⚙️  Processing", config.processing),
            ("jira_matching", "🎫 JIRA Matching", config.jira_matching)
        ]
        
        all_passed = True
        
        for need_key, need_title, assignment in needs:
            self.console.print(f"\n[yellow]Testing {need_title}...[/yellow]")
            self.console.print(f"  Provider: {assignment.provider}")
            self.console.print(f"  Model: {assignment.model}")
            
            try:
                success = llm_client.call_llm(need_key, [{"role": "user", "content": "Test - respond with 'OK'"}], max_tokens=10)
                if "OK" in success.upper() or "ok" in success.lower():
                    self.console.print(f"  [green]✅ {need_title} working![/green]")
                else:
                    self.console.print(f"  [green]✅ {need_title} responding (got: {success.strip()[:20]}...)[/green]")
            except Exception as e:
                self.console.print(f"  [red]❌ {need_title} failed: {str(e)[:50]}...[/red]")
                all_passed = False
        
        if all_passed:
            self.console.print(f"\n[bold green]🎉 Configuration '{config_name}' is working perfectly![/bold green]")
        else:
            self.console.print(f"\n[bold yellow]⚠️  Configuration '{config_name}' has some issues.[/bold yellow]")
            self.console.print("Check your providers and API keys.")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _show_simple_model_settings_help(self):
        """Show simple help for the Model Settings menu."""
        self.console.clear()
        self.console.print("\n[bold cyan]📖 Model Settings Help[/bold cyan]")
        
        help_content = """
[bold]Model Settings Overview:[/bold]

The Work Journal uses AI models for 3 different needs:
  💬 [green]Conversation[/green] - Interactive chat and refinement
  ⚙️  [green]Processing[/green] - Structure and analyze work entries  
  🎫 [green]JIRA Matching[/green] - Find relevant JIRA tickets

[bold]Menu Options:[/bold]

[green]1. Switch Configuration[/green] 🔄
   Switch between your saved configurations.
   Each configuration assigns specific models to the 3 needs.

[green]2. Create New Configuration[/green] ➕
   Build a new configuration using the 4-step process:
   • Step 1: Add Providers (connect to AI services)
   • Step 2: Assign Models (pick models for each need)
   • Step 3: Name Configuration (give it a memorable name)
   • Step 4: Activate (set as current configuration)

[green]3. Test Current Configuration[/green] 🧪
   Test all 3 needs in your current configuration to make
   sure everything is working properly.

[bold]Provider Types:[/bold]
  💻 [green]Local[/green] - LM Studio, Ollama (free, private)
  ☁️  [green]Cloud[/green] - OpenAI, Anthropic (paid, requires API key)
  🔧 [green]Custom[/green] - Other OpenAI-compatible endpoints

[dim]Press Enter to go back...[/dim]"""
        
        self.console.print(help_content)
        self.safe_input()
    
    def _show_model_settings_help(self):
        """Show comprehensive help for Model Settings menu."""
        self.console.clear()
        self.console.print("\n[bold cyan]📖 Model Settings Help[/bold cyan]")
        
        help_content = """
[bold]Understanding the Model Settings Menu:[/bold]

[green]1. Change Config[/green] 🔄
   Switch between pre-built configurations like:
   • lmstudio_fast - Fast local models
   • cloud_openai - OpenAI GPT models  
   • ollama_quality - High-quality local models
   • cloud_anthropic - Claude models
   
   [dim]Use this when: You want to quickly switch between different AI setups[/dim]

[green]2. Providers[/green] 🏢
   Set up AI service providers:
   • Local: LM Studio, Ollama (free, private, runs on your computer)
   • Cloud: OpenAI, Anthropic (paid, requires API keys, more powerful)
   
   [dim]Use this when: Setting up for the first time or adding new AI services[/dim]

[green]3. Models[/green] 🤖
   Browse available models from your configured providers:
   • See what models each provider offers
   • Check model availability and status
   
   [dim]Use this when: You want to see what AI models you can use[/dim]

[green]4. Configs[/green] ⚙️
   Create custom configurations:
   • Combine specific providers + models for different tasks
   • Set up workflows (conversation vs processing vs matching)
   
   [dim]Use this when: Pre-built configs don't fit your needs[/dim]

[green]5. Test[/green] 🧪
   Test your current setup:
   • Verify providers are working
   • Check model connectivity
   • Troubleshoot issues
   
   [dim]Use this when: Something isn't working or you want to verify setup[/dim]

[bold yellow]Quick Start Guide:[/bold yellow]
1. First time? Start with [green]Providers[/green] to set up your AI services
2. Then use [green]Change Config[/green] to pick a pre-built setup
3. Use [green]Test[/green] to make sure everything works
4. Need something custom? Try [green]Configs[/green] to create your own
        """
        
        self.console.print(Panel(help_content.strip(), border_style="cyan", padding=(1, 2)))
        self.console.print("\n[dim]Press Enter to return to menu...[/dim]")
        self.safe_input()
    
    def _quick_setup_wizard(self, llm_client: 'LLMClient'):
        """Quick setup wizard for new users."""
        self.console.clear()
        self.console.print("\n[bold green]🚀 Quick Setup Wizard[/bold green]")
        self.console.print("Let's get you set up with AI models quickly!")
        
        self.console.print("\n[bold]Step 1: Choose your preferred setup[/bold]")
        setup_options = """
  [green]1.[/green] 💻 Local models (free, private, slower)
      Uses models running on your computer via LM Studio or Ollama
      
  [green]2.[/green] ☁️  Cloud models (paid, fast, powerful) 
      Uses OpenAI or Anthropic APIs (requires API key)
      
  [green]3.[/green] 🔄 Hybrid (recommended)
      Local for conversations, cloud for processing
        """
        
        self.console.print(setup_options)
        self.console.print("\n[cyan bold]Setup choice (1-3)>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if choice == '1':
            self._setup_local_only(llm_client)
        elif choice == '2':
            self._setup_cloud_only(llm_client)
        elif choice == '3':
            self._setup_hybrid(llm_client)
        else:
            self.console.print("[yellow]Setup cancelled.[/yellow]")
    
    def _setup_local_only(self, llm_client: 'LLMClient'):
        """Set up local-only configuration."""
        self.console.print("\n[bold cyan]💻 Local Setup[/bold cyan]")
        
        config = llm_client.config
        
        # Find available local providers
        local_providers = []
        local_presets = []
        
        for provider_name, provider_config in config.providers.items():
            if "localhost" in provider_config.api_base:
                local_providers.append(provider_name)
        
        # Find presets that use local providers
        for preset_name, preset_config in config.presets.items():
            processing_provider = preset_config["processing"].provider
            if processing_provider in local_providers:
                local_presets.append((preset_name, processing_provider))
        
        if not local_providers:
            self.console.print("[yellow]No local providers configured yet.[/yellow]")
            self.console.print("You need to set up a local provider first (LM Studio, Ollama, etc.)")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        if not local_presets:
            self.console.print("[yellow]No local configurations available.[/yellow]")
            self.console.print("Available local providers but no presets configured for them.")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        self.console.print("Choose your local configuration:")
        
        for i, (preset_name, provider_name) in enumerate(local_presets, 1):
            provider_title = provider_name.replace('_', ' ').title()
            self.console.print(f"  [green]{i}.[/green] {provider_title} ({preset_name})")
        
        self.console.print(f"\n[cyan bold]Local config (1-{len(local_presets)})>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(local_presets):
                preset_name, provider_name = local_presets[choice_num - 1]
                self.switch_to_preset(llm_client, config, preset_name)
                
                provider_title = provider_name.replace('_', ' ').title()
                self.console.print(f"\n[green]✅ Configured for {provider_title}![/green]")
                
                # Give helpful instructions based on provider
                if "lmstudio" in provider_name.lower():
                    self.console.print("[dim]Make sure LM Studio is running with a model loaded.[/dim]")
                elif "ollama" in provider_name.lower():
                    self.console.print("[dim]Make sure Ollama is running: ollama serve[/dim]")
                else:
                    self.console.print(f"[dim]Make sure {provider_title} is running and accessible.[/dim]")
            else:
                self.console.print(f"[red]Invalid choice. Select 1-{len(local_presets)}[/red]")
        except ValueError:
            self.console.print("[red]Invalid input. Enter a number.[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _setup_cloud_only(self, llm_client: 'LLMClient'):
        """Set up cloud-only configuration."""
        self.console.print("\n[bold cyan]☁️  Cloud Setup[/bold cyan]")
        
        config = llm_client.config
        
        # Find available cloud providers and presets
        cloud_options = []
        
        # Check for OpenAI provider and presets
        if "openai" in config.providers:
            for preset_name, preset_config in config.presets.items():
                if preset_config["processing"].provider == "openai":
                    cloud_options.append(("OpenAI", "openai", preset_name, "GPT models"))
                    break
        
        # Check for Anthropic provider and presets
        if "anthropic" in config.providers:
            for preset_name, preset_config in config.presets.items():
                if preset_config["processing"].provider == "anthropic":
                    cloud_options.append(("Anthropic", "anthropic", preset_name, "Claude models"))
                    break
        
        # Check for other cloud providers (non-localhost)
        for provider_name, provider_config in config.providers.items():
            if "localhost" not in provider_config.api_base and provider_name not in ["openai", "anthropic"]:
                # Find presets for this provider
                for preset_name, preset_config in config.presets.items():
                    if preset_config["processing"].provider == provider_name:
                        cloud_options.append((provider_name.title(), provider_name, preset_name, "Custom cloud"))
                        break
        
        if not cloud_options:
            self.console.print("[yellow]No cloud providers configured yet.[/yellow]")
            self.console.print("You need to set up a cloud provider first (OpenAI, Anthropic, etc.)")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        self.console.print("Choose your cloud provider:")
        
        for i, (display_name, provider_name, preset_name, description) in enumerate(cloud_options, 1):
            self.console.print(f"  [green]{i}.[/green] {display_name} ({description})")
        
        self.console.print(f"\n[cyan bold]Cloud provider (1-{len(cloud_options)})>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(cloud_options):
                display_name, provider_name, preset_name, description = cloud_options[choice_num - 1]
                
                # Check API key setup for known providers
                if provider_name == "openai":
                    if not self._setup_openai_keys(llm_client) or not self._has_openai_key():
                        return
                elif provider_name == "anthropic":
                    if not self._setup_anthropic_keys(llm_client) or not self._has_anthropic_key():
                        return
                
                # Switch to the preset
                self.switch_to_preset(llm_client, config, preset_name)
                self.console.print(f"\n[green]✅ Configured for {display_name}![/green]")
            else:
                self.console.print(f"[red]Invalid choice. Select 1-{len(cloud_options)}[/red]")
        except ValueError:
            self.console.print("[red]Invalid input. Enter a number.[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _setup_hybrid(self, llm_client: 'LLMClient'):
        """Set up hybrid local + cloud configuration."""
        self.console.print("\n[bold cyan]🔄 Hybrid Setup[/bold cyan]")
        self.console.print("This uses local models for conversation and cloud for processing.")
        self.console.print("You'll need both a local provider AND a cloud API key.")
        
        # Set up cloud for processing first
        self.console.print("\n[bold]Step 1: Cloud provider for processing[/bold]")
        self.console.print("  [green]1.[/green] OpenAI")
        self.console.print("  [green]2.[/green] Anthropic")
        
        self.console.print("\n[cyan bold]Cloud provider (1-2)>[/cyan bold] ", end="")
        cloud_choice = self.safe_input()
        
        # Set up local for conversation
        self.console.print("\n[bold]Step 2: Local provider for conversation[/bold]")
        self.console.print("  [green]1.[/green] LM Studio")
        self.console.print("  [green]2.[/green] Ollama")
        
        self.console.print("\n[cyan bold]Local provider (1-2)>[/cyan bold] ", end="")
        local_choice = self.safe_input()
        
        # Configure based on choices
        if cloud_choice == '1' and self._setup_openai_keys(llm_client) and self._has_openai_key():
            if local_choice == '1':
                # Custom hybrid: OpenAI processing + LM Studio conversation
                self._create_hybrid_config(llm_client, "openai", "gpt-4o", "lmstudio", "llama-3.2-3b-instruct")
            else:
                # Custom hybrid: OpenAI processing + Ollama conversation  
                self._create_hybrid_config(llm_client, "openai", "gpt-4o", "ollama", "llama3.2:3b")
                
        elif cloud_choice == '2' and self._setup_anthropic_keys(llm_client) and self._has_anthropic_key():
            if local_choice == '1':
                # Custom hybrid: Anthropic processing + LM Studio conversation
                self._create_hybrid_config(llm_client, "anthropic", "claude-3-5-sonnet-20241022", "lmstudio", "llama-3.2-3b-instruct")
            else:
                # Custom hybrid: Anthropic processing + Ollama conversation
                self._create_hybrid_config(llm_client, "anthropic", "claude-3-5-sonnet-20241022", "ollama", "llama3.2:3b")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _setup_openai_keys(self, llm_client: 'LLMClient') -> bool:
        """Help user set up OpenAI API key."""
        self.console.print("\n[bold yellow]🔑 OpenAI API Key Setup[/bold yellow]")
        
        if self._has_openai_key():
            self.console.print("[green]✅ OpenAI API key is already configured![/green]")
            return True
        
        self.console.print("You need an OpenAI API key to use GPT models.")
        self.console.print("[dim]Get one at: https://platform.openai.com/api-keys[/dim]")
        
        env_file = llm_client.storage.ensure_env_file()
        self.console.print(f"\nPlease add your API key to: [cyan]{env_file}[/cyan]")
        self.console.print("Add this line: [yellow]OPENAI_API_KEY=your_key_here[/yellow]")
        
        self.console.print("\n[dim]Press Enter when you've added your API key...[/dim]")
        self.safe_input()
        
        return self._has_openai_key()
    
    def _setup_anthropic_keys(self, llm_client: 'LLMClient') -> bool:
        """Help user set up Anthropic API key."""
        self.console.print("\n[bold yellow]🔑 Anthropic API Key Setup[/bold yellow]")
        
        if self._has_anthropic_key():
            self.console.print("[green]✅ Anthropic API key is already configured![/green]")
            return True
        
        self.console.print("You need an Anthropic API key to use Claude models.")
        self.console.print("[dim]Get one at: https://console.anthropic.com/[/dim]")
        
        env_file = llm_client.storage.ensure_env_file()
        self.console.print(f"\nPlease add your API key to: [cyan]{env_file}[/cyan]") 
        self.console.print("Add this line: [yellow]ANTHROPIC_API_KEY=your_key_here[/yellow]")
        
        self.console.print("\n[dim]Press Enter when you've added your API key...[/dim]")
        self.safe_input()
        
        return self._has_anthropic_key()
    
    def _has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def _has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        import os
        from dotenv import load_dotenv  
        load_dotenv()
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    
    def _create_hybrid_config(self, llm_client: 'LLMClient', 
                            cloud_provider: str, cloud_model: str,
                            local_provider: str, local_model: str):
        """Create a custom hybrid configuration."""
        from .models import WorkflowSettings
        
        config = llm_client.config
        
        # Create hybrid workflows
        config.workflows = {
            "processing": WorkflowSettings(provider=cloud_provider, model=cloud_model),
            "conversation": WorkflowSettings(provider=local_provider, model=local_model),
            "jira_matching": WorkflowSettings(provider=local_provider, model=local_model)
        }
        
        # Save as new preset
        preset_name = f"hybrid_{cloud_provider}_{local_provider}"
        config.presets[preset_name] = {
            "processing": config.workflows["processing"],
            "conversation": config.workflows["conversation"],
            "jira_matching": config.workflows["jira_matching"]
        }
        config.current_preset = preset_name
        
        llm_client.storage.save_config(config)
        self.console.print(f"\n[green]✅ Created hybrid configuration: {preset_name}[/green]")
        self.console.print(f"[dim]{cloud_provider.title()} for processing, {local_provider.title()} for conversation[/dim]")
    
    def _switch_configuration_menu(self, llm_client: 'LLMClient', config):
        """Show configuration switching menu."""
        self.console.clear()
        self.console.print("\n[bold cyan]🔄 Switch Configuration[/bold cyan]")
        
        # Show current configuration
        current_display = config.current_preset if config.current_preset else "[red]None[/red]"
        self.console.print(f"[dim]Current: {current_display}[/dim]")
        
        # Check if there are any configurations available
        presets = list(config.presets.keys())
        if not presets:
            self.console.print("\n[yellow]No configurations available![/yellow]")
            self.console.print("You need to create a configuration first.")
            self.console.print("\nTo get started:")
            self.console.print("  1. Go back to Model Settings menu")
            self.console.print("  2. Choose [green]2. Providers[/green] to set up AI services")
            self.console.print("  3. Then choose [green]4. Configs[/green] to create a configuration")
            self.console.print("\n[dim]Press Enter to go back...[/dim]")
            self.safe_input()
            return
        
        # Show available configurations
        self.console.print("\n[bold]Available Configurations:[/bold]")
        for i, preset_name in enumerate(presets, 1):
            preset = config.presets[preset_name]
            current_marker = " [green]← current[/green]" if preset_name == config.current_preset else ""
            
            self.console.print(f"  [cyan]{i}.[/cyan] {preset_name}{current_marker}")
            self.console.print(f"      Processing: {preset['processing'].provider} / {preset['processing'].model}")
        
        self.console.print(f"\n[cyan bold]Switch to (1-{len(presets)}) or 'q' to go back>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if choice is None or choice.lower() == 'q':
            return  # Go back to previous menu
        
        try:
            preset_num = int(choice)
            if 1 <= preset_num <= len(presets):
                new_preset = presets[preset_num - 1]
                if new_preset != config.current_preset:
                    self.switch_to_preset(llm_client, config, new_preset)
                else:
                    self.console.print("[yellow]Already using that configuration.[/yellow]")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    self.safe_input()
            else:
                self.console.print(f"[red]Invalid selection. Choose 1-{len(presets)} or 'q' to go back.[/red]")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
        except ValueError:
            self.console.print("[red]Invalid input. Enter a number or 'q' to go back.[/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
    
    def _show_configuration_details(self, llm_client: 'LLMClient', config):
        """Show detailed configuration information."""
        self.console.clear()
        self.console.print("\n[bold cyan]📊 Configuration Details[/bold cyan]")
        
        # Show current configuration
        current_workflows = config.workflows
        settings_text = f"""
[bold]Current Configuration: {config.current_preset}[/bold]
  Processing: {current_workflows['processing'].provider} / {current_workflows['processing'].model}
  Conversation: {current_workflows['conversation'].provider} / {current_workflows['conversation'].model}  
  Jira Matching: {current_workflows['jira_matching'].provider} / {current_workflows['jira_matching'].model}
        """
        
        self.console.print(Panel(settings_text.strip(), title="Active Models", border_style="cyan"))
        
        # Show all available configurations
        self.console.print("\n[bold]All Available Configurations:[/bold]")
        presets = list(config.presets.keys())
        for i, preset_name in enumerate(presets, 1):
            preset = config.presets[preset_name]
            current_marker = " [green]← current[/green]" if preset_name == config.current_preset else ""
            
            self.console.print(f"  [cyan]{i}.[/cyan] {preset_name}{current_marker}")
            self.console.print(f"      Processing: {preset['processing'].provider} / {preset['processing'].model}")
        
        # Show provider information
        self.console.print("\n[bold]Configured Providers:[/bold]")
        for provider_name, provider_config in config.providers.items():
            auth_info = "API Key" if provider_config.api_key else f"ENV: {provider_config.auth_env}" if provider_config.auth_env else "No Auth"
            self.console.print(f"  [yellow]{provider_name.title()}[/yellow]: {provider_config.api_base} ({auth_info})")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def browse_models(self, llm_client: 'LLMClient'):
        """Browse available models from different providers."""
        # Use actual configured providers instead of hardcoded list
        config = llm_client.config
        providers = list(config.providers.keys())
        
        while True:
            # Clear screen for each menu display
            self.console.clear()
            
            self.console.print(f"\n[bold cyan]📋 Browse Models[/bold cyan]")
            
            if not providers:
                self.console.print("[yellow]No providers configured yet.[/yellow]")
                self.console.print("Go to Model Settings → Set up a new AI provider first.")
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
                return
            
            # Show providers with helpful info
            self.console.print("\n[bold]Select Provider:[/bold]")
            for i, provider_name in enumerate(providers, 1):
                provider_config = config.providers[provider_name]
                # Show endpoint type for context
                if "localhost" in provider_config.api_base:
                    location_info = "[dim](Local)[/dim]"
                elif "openai.com" in provider_config.api_base:
                    location_info = "[dim](OpenAI Cloud)[/dim]"
                elif "anthropic.com" in provider_config.api_base:
                    location_info = "[dim](Anthropic Cloud)[/dim]"
                else:
                    location_info = "[dim](Custom)[/dim]"
                
                self.console.print(f"  [cyan]{i}.[/cyan] {provider_name.title()} {location_info}")
            
            self.console.print(f"\n[dim]Options: 1-{len(providers)} to browse provider, [yellow]q[/yellow]=back[/dim]")
            
            self.console.print("\n[cyan bold]BROWSE>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            
            try:
                provider_num = int(choice)
                if 1 <= provider_num <= len(providers):
                    provider = providers[provider_num - 1]
                    self.show_provider_models(llm_client, provider)
                else:
                    self.console.print(f"[red]Invalid selection. Choose 1-{len(providers)}[/red]")
            except ValueError:
                self.console.print("[red]Invalid input. Enter a number or 'q'.[/red]")
    
    def show_provider_models(self, llm_client: 'LLMClient', provider: str):
        """Show available models for a specific provider."""
        self.console.print(f"\n[bold yellow]🔍 {provider.title()} Models[/bold yellow]")
        
        # Get available models
        self.console.print("[dim]🤖⏳ Fetching available models...[/dim]")
        models = llm_client.get_available_models(provider)
        
        if not models:
            self.console.print(f"[red]❌ No models found for {provider}[/red]")
            if provider == "lmstudio":
                self.console.print("[yellow]💡 Make sure LM Studio is running with models loaded[/yellow]")
            elif provider == "ollama":
                self.console.print("[yellow]💡 Make sure Ollama is running: `ollama serve`[/yellow]")
            else:
                self.console.print(f"[yellow]💡 Make sure your {provider.upper()}_API_KEY is set[/yellow]")
            
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        # Display models
        self.console.print(f"\n[bold]Available Models ({len(models)} found):[/bold]")
        for i, model in enumerate(models, 1):
            self.console.print(f"  [cyan]{i:2d}.[/cyan] {model}")
        
        self.console.print(f"\n[dim]Options: 1-{len(models)} to use model, [yellow]t[/yellow]=test model, [yellow]q[/yellow]=back[/dim]")
        
        while True:
            self.console.print(f"\n[cyan bold]{provider.upper()}>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            choice = choice.lower()
            
            if choice == 't':
                # Test a specific model
                self.console.print("Enter model number to test: ", end="")
                test_choice = self.safe_input()
                if test_choice and test_choice.isdigit():
                    model_num = int(test_choice)
                    if 1 <= model_num <= len(models):
                        model = models[model_num - 1]
                        self.console.print(f"[dim]🤖⏳ Testing {provider}/{model}...[/dim]")
                        success, error_msg = llm_client.test_provider(provider, model)
                        if success:
                            self.console.print(f"[green]✅ {model} works![/green]")
                        else:
                            self.console.print(f"[red]❌ {model} failed[/red]")
            else:
                # Try to use a specific model
                try:
                    model_num = int(choice)
                    if 1 <= model_num <= len(models):
                        model = models[model_num - 1]
                        if self.use_model_for_config(llm_client, provider, model):
                            self.console.print(f"[green]✅ Now using {provider}/{model}[/green]")
                            break
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(models)}[/red]")
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number, 't', or 'q'.[/red]")
    
    def use_model_for_config(self, llm_client: 'LLMClient', provider: str, model: str) -> bool:
        """Create a temporary preset using the selected model."""
        self.console.print(f"\n[bold]🔧 Configure {provider}/{model}[/bold]")
        self.console.print("Apply this model to:")
        self.console.print("  [cyan]1.[/cyan] Processing only")
        self.console.print("  [cyan]2.[/cyan] All workflows (processing, conversation, jira)")
        self.console.print("  [cyan]3.[/cyan] Create new configuration")
        
        self.console.print("\n[cyan bold]APPLY>[/cyan bold] ", end="")
        choice = self.safe_input()
        if choice is None:
            return False
            
        from .models import WorkflowSettings
        
        try:
            if choice == '1':
                # Update processing workflow only
                llm_client.config.workflows["processing"] = WorkflowSettings(provider=provider, model=model)
                llm_client.storage.save_config(llm_client.config)
                return True
            elif choice == '2':
                # Update all workflows
                workflow = WorkflowSettings(provider=provider, model=model)
                llm_client.config.workflows = {
                    "processing": workflow,
                    "conversation": workflow,
                    "jira_matching": workflow
                }
                llm_client.storage.save_config(llm_client.config)
                return True
            elif choice == '3':
                # Create new preset
                return self.create_preset_with_model(llm_client, provider, model)
            else:
                self.console.print("[red]Invalid choice.[/red]")
                return False
        except Exception as e:
            self.console.print(f"[red]Error applying configuration: {e}[/red]")
            return False
    
    def create_preset_with_model(self, llm_client: 'LLMClient', provider: str, model: str) -> bool:
        """Create a new configuration with the selected model."""
        self.console.print(f"\n[bold]📝 Create New Configuration[/bold]")
        self.console.print(f"[dim]Using: {provider}/{model}[/dim]")
        
        self.console.print("\nEnter configuration name: ", end="")
        preset_name = self.safe_input()
        if not preset_name:
            return False
            
        # Check if preset exists
        if preset_name in llm_client.config.presets:
            self.console.print(f"[yellow]Configuration '{preset_name}' already exists. Overwrite? (y/n): [/yellow]", end="")
            confirm = self.safe_input()
            if confirm is None or confirm.lower() != 'y':
                return False
        
        from .models import WorkflowSettings
        
        # Create the preset
        workflow = WorkflowSettings(provider=provider, model=model)
        llm_client.config.presets[preset_name] = {
            "processing": workflow,
            "conversation": workflow,
            "jira_matching": workflow
        }
        
        # Switch to new preset
        llm_client.config.current_preset = preset_name
        llm_client.config.workflows = {
            "processing": workflow,
            "conversation": workflow,
            "jira_matching": workflow
        }
        
        llm_client.storage.save_config(llm_client.config)
        self.console.print(f"[green]✅ Created and switched to configuration: {preset_name}[/green]")
        return True
    
    def create_custom_config(self, llm_client: 'LLMClient') -> bool:
        """Create a custom configuration interactively."""
        
        # Show configuration options
        while True:
            # Clear screen for each menu display
            self.console.clear()
            
            self.console.print(f"\n[bold cyan]🛠️  Custom Configuration[/bold cyan]")
            config_text = """
[bold]Configuration Options:[/bold]
  [green]1.[/green] Configure workflows (assign models to different tasks)
  [green]2.[/green] Manage providers (add/edit API endpoints)  
  [green]3.[/green] Create new preset (save current config)
  [green]4.[/green] Quick setup templates (local, hybrid, cloud)
  [green]5.[/green] Import/Export configuration
  [green]h.[/green] Help with configuration
  [green]q.[/green] Back to model settings
            """
            
            self.console.print(config_text)
            self.console.print("\n[cyan bold]CONFIG>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                return False
            choice = choice.lower()
            
            if choice == '1':
                if self.configure_workflows(llm_client):
                    return True  # Configuration changed
            elif choice == '2':
                if self.manage_providers(llm_client):
                    return True  # Configuration changed
            elif choice == '3':
                if self.create_preset(llm_client):
                    return True  # Configuration changed
            elif choice == '4':
                if self.setup_templates(llm_client):
                    return True  # Configuration changed
            elif choice == '5':
                self.import_export_config(llm_client)
            elif choice == 'h':
                self.show_config_help()
            elif choice == '':
                # Empty command, just redraw menu
                pass
            else:
                self.console.print(f"[red]Unknown command: '{choice}'. Use numbers or letters from the menu.[/red]")
    
    def configure_workflows(self, llm_client: 'LLMClient') -> bool:
        """Configure models for different workflows."""
        self.console.print(f"\n[bold yellow]⚙️  Configure Workflows[/bold yellow]")
        self.console.print("Assign different models to different tasks for optimal performance.")
        
        workflows = ["processing", "conversation", "jira_matching"]
        workflow_descriptions = {
            "processing": "Extracts structured data from journal entries",
            "conversation": "Handles interactive refinement and questions", 
            "jira_matching": "Matches work to JIRA tickets"
        }
        
        config = llm_client.config
        changes_made = False
        
        while True:
            # Show current workflow assignments
            self.console.print(f"\n[bold]Current Workflow Assignments:[/bold]")
            for i, workflow in enumerate(workflows, 1):
                current = config.workflows[workflow]
                fallback_text = ""
                if current.fallback:
                    fallback_text = f" (fallback: {current.fallback['provider']}/{current.fallback['model']})"
                
                self.console.print(f"  [cyan]{i}.[/cyan] [bold]{workflow.replace('_', ' ').title()}[/bold]")
                self.console.print(f"      {workflow_descriptions[workflow]}")
                self.console.print(f"      [green]→ {current.provider}/{current.model}[/green]{fallback_text}")
            
            # Show options
            self.console.print(f"\n[dim]Options: 1-{len(workflows)} to configure workflow, [yellow]s[/yellow]=save changes, [yellow]q[/yellow]=back[/dim]")
            self.console.print("\n[cyan bold]WORKFLOW>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            choice = choice.lower()
            
            if choice == 's':
                if changes_made:
                    llm_client.storage.save_config(config)
                    self.console.print("[green]✅ Workflow configuration saved[/green]")
                    return True
                else:
                    self.console.print("[yellow]No changes to save[/yellow]")
            else:
                # Try to parse as workflow number
                try:
                    workflow_num = int(choice)
                    if 1 <= workflow_num <= len(workflows):
                        workflow_name = workflows[workflow_num - 1]
                        if self.configure_single_workflow(llm_client, workflow_name):
                            changes_made = True
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(workflows)}[/red]")
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number, 's', or 'q'.[/red]")
        
        return changes_made

    def configure_single_workflow(self, llm_client: 'LLMClient', workflow_name: str) -> bool:
        """Configure a single workflow's model assignment."""
        self.console.print(f"\n[bold green]🎯 Configure {workflow_name.replace('_', ' ').title()}[/bold green]")
        
        # Get available providers and models
        config = llm_client.config
        current_setting = config.workflows[workflow_name]
        
        # Show current setting
        self.console.print(f"Current: [yellow]{current_setting.provider}/{current_setting.model}[/yellow]")
        
        # Get provider choice
        providers = list(config.providers.keys())
        self.console.print(f"\n[bold]Available Providers:[/bold]")
        for i, provider in enumerate(providers, 1):
            status = "●" if provider == current_setting.provider else "○"
            self.console.print(f"  [cyan]{i}.[/cyan] {status} {provider.title()}")
        
        self.console.print(f"\n[dim]Select provider (1-{len(providers)}) or Enter to keep current:[/dim]")
        self.console.print("[green]Provider>[/green] ", end="")
        provider_choice = self.safe_input()
        if provider_choice is None:
            return False
        
        # Parse provider choice
        if not provider_choice:
            # Keep current provider
            selected_provider = current_setting.provider
        else:
            try:
                provider_num = int(provider_choice)
                if 1 <= provider_num <= len(providers):
                    selected_provider = providers[provider_num - 1]
                else:
                    self.console.print(f"[red]Invalid provider selection[/red]")
                    return False
            except ValueError:
                self.console.print(f"[red]Invalid input[/red]")
                return False
        
        # Get available models for selected provider
        self.console.print(f"\n[dim]🤖⏳ Fetching models for {selected_provider}...[/dim]")
        available_models = llm_client.get_available_models(selected_provider)
        
        if not available_models:
            self.console.print(f"[red]❌ No models available for {selected_provider}[/red]")
            if selected_provider in ["lmstudio", "ollama"]:
                self.console.print(f"[yellow]💡 Make sure {selected_provider} is running with models loaded[/yellow]")
            else:
                self.console.print(f"[yellow]💡 Check your {selected_provider.upper()}_API_KEY environment variable[/yellow]")
            return False
        
        # Show model choices
        self.console.print(f"\n[bold]Available Models for {selected_provider.title()}:[/bold]")
        for i, model in enumerate(available_models, 1):
            status = "●" if model == current_setting.model and selected_provider == current_setting.provider else "○"
            self.console.print(f"  [cyan]{i}.[/cyan] {status} {model}")
        
        self.console.print(f"\n[dim]Select model (1-{len(available_models)}) or Enter to keep current:[/dim]")
        self.console.print("[green]Model>[/green] ", end="")
        model_choice = self.safe_input()
        if model_choice is None:
            return False
        
        # Parse model choice
        if not model_choice:
            # Keep current model if same provider, otherwise use first available
            if selected_provider == current_setting.provider:
                selected_model = current_setting.model
            else:
                selected_model = available_models[0]
        else:
            try:
                model_num = int(model_choice)
                if 1 <= model_num <= len(available_models):
                    selected_model = available_models[model_num - 1]
                else:
                    self.console.print(f"[red]Invalid model selection[/red]")
                    return False
            except ValueError:
                self.console.print(f"[red]Invalid input[/red]")
                return False
        
        # Update workflow configuration
        from .models import WorkflowSettings
        config.workflows[workflow_name] = WorkflowSettings(
            provider=selected_provider,
            model=selected_model,
            fallback=current_setting.fallback  # Keep existing fallback
        )
        
        self.console.print(f"[green]✅ {workflow_name.replace('_', ' ').title()} updated: {selected_provider}/{selected_model}[/green]")
        return True

    def create_preset(self, llm_client: 'LLMClient') -> bool:
        """Create a new preset from current configuration."""
        self.console.print(f"\n[bold green]💾 Create New Preset[/bold green]")
        self.console.print("Save your current configuration as a named preset for easy switching.")
        
        config = llm_client.config
        
        # Show current configuration
        self.console.print(f"\n[bold]Current Configuration:[/bold]")
        for workflow_name, workflow_config in config.workflows.items():
            self.console.print(f"  {workflow_name.replace('_', ' ').title()}: [yellow]{workflow_config.provider}/{workflow_config.model}[/yellow]")
        
        # Get preset name
        self.console.print(f"\n[dim]Enter a name for this preset:[/dim]")
        self.console.print("[green]Preset name>[/green] ", end="")
        preset_name = self.safe_input()
        if preset_name is None or not preset_name.strip():
            self.console.print("[yellow]Preset creation cancelled.[/yellow]")
            return False
        
        preset_name = preset_name.strip()
        
        # Check if preset already exists
        if preset_name in config.presets:
            self.console.print(f"[yellow]Preset '{preset_name}' already exists. Overwrite? (y/N)>[/yellow] ", end="")
            confirm = self.safe_input()
            if confirm is None or confirm.lower() != 'y':
                self.console.print("[yellow]Preset creation cancelled.[/yellow]")
                return False
        
        # Validate configuration before saving
        if self.validate_configuration(llm_client, silent=True):
            validation_status = "[green]✅ Configuration validated[/green]"
        else:
            validation_status = "[yellow]⚠️  Configuration has issues (saved anyway)[/yellow]"
        
        # Create preset from current workflows
        config.presets[preset_name] = {
            workflow_name: workflow_config.model_copy()
            for workflow_name, workflow_config in config.workflows.items()
        }
        
        # Save configuration  
        llm_client.storage.save_config(config)
        self.console.print(f"[green]✅ Preset '{preset_name}' created successfully[/green]")
        self.console.print(validation_status)
        
        # Ask if they want to switch to this preset
        self.console.print(f"[dim]Set '{preset_name}' as active preset? (y/N)>[/dim] ", end="")
        switch_confirm = self.safe_input()
        if switch_confirm and switch_confirm.lower() == 'y':
            config.current_preset = preset_name
            llm_client.storage.save_config(config)
            self.console.print(f"[green]✅ Switched to preset '{preset_name}'[/green]")
        
        return True

    def delete_preset_interactive(self, llm_client: 'LLMClient') -> bool:
        """Interactive configuration deletion with safety checks."""
        self.console.print(f"\n[bold red]🗑️  Delete Configuration[/bold red]")
        
        config = llm_client.config
        presets = list(config.presets.keys())
        
        if not presets:
            self.console.print("[yellow]No configurations to delete.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        if len(presets) == 1:
            self.console.print("[yellow]Cannot delete the only remaining configuration.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        # Show available configurations
        self.console.print(f"\n[bold red]⚠️  WARNING: Deleting a configuration cannot be undone![/bold red]")
        self.console.print(f"\n[bold]Available Configurations:[/bold]")
        for i, preset_name in enumerate(presets, 1):
            current_marker = " [red]← CURRENT (cannot delete)[/red]" if preset_name == config.current_preset else ""
            self.console.print(f"  [cyan]{i}.[/cyan] {preset_name.title()}{current_marker}")
        
        self.console.print(f"\n[dim]Select preset to delete (1-{len(presets)}) or q to cancel:[/dim]")
        self.console.print("\n[red bold]DELETE>[/red bold] ", end="")
        choice = self.safe_input()
        
        if choice is None or choice.lower() == 'q':
            return False
        
        try:
            preset_num = int(choice)
            if 1 <= preset_num <= len(presets):
                preset_name = presets[preset_num - 1]
                return self.confirm_preset_deletion(llm_client, preset_name)
            else:
                self.console.print(f"[red]Invalid selection. Choose 1-{len(presets)}[/red]")
                return False
        except ValueError:
            self.console.print("[red]Invalid input. Enter a number or 'q'.[/red]")
            return False

    def confirm_preset_deletion(self, llm_client: 'LLMClient', preset_name: str) -> bool:
        """Confirm preset deletion with safety checks."""
        config = llm_client.config
        
        # Prevent deletion of current preset
        if preset_name == config.current_preset:
            self.console.print(f"[red]Cannot delete the currently active preset '{preset_name}'.[/red]")
            self.console.print("[yellow]Switch to a different preset first, then delete this one.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        # Show preset details
        preset_config = config.presets[preset_name]
        self.console.print(f"\n[bold red]🚨 Confirm Deletion: '{preset_name}'[/bold red]")
        
        self.console.print(f"\n[bold]Preset Configuration:[/bold]")
        for workflow_name, workflow_config in preset_config.items():
            fallback_text = ""
            if workflow_config.fallback:
                fallback_text = f" (fallback: {workflow_config.fallback['provider']}/{workflow_config.fallback['model']})"
            self.console.print(f"  {workflow_name.replace('_', ' ').title()}: [yellow]{workflow_config.provider}/{workflow_config.model}[/yellow]{fallback_text}")
        
        # Final confirmation
        self.console.print(f"\n[bold red]This action cannot be undone![/bold red]")
        self.console.print(f"[red bold]Type 'DELETE {preset_name.upper()}' to confirm>[/red bold] ", end="")
        confirmation = self.safe_input()
        
        if confirmation == f"DELETE {preset_name.upper()}":
            # Remove preset
            del config.presets[preset_name]
            
            # Save configuration
            llm_client.storage.save_config(config)
            
            self.console.print(f"\n[green]✅ Preset '{preset_name}' deleted successfully[/green]")
            
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return True
        else:
            self.console.print(f"[yellow]Deletion cancelled - confirmation did not match[/yellow]")
            return False

    def edit_preset_interactive(self, llm_client: 'LLMClient') -> bool:
        """Interactive preset editing with options to rename or modify configuration."""
        self.console.print(f"\n[bold cyan]✏️  Edit Preset[/bold cyan]")
        
        config = llm_client.config
        presets = list(config.presets.keys())
        
        if not presets:
            self.console.print("[yellow]No presets to edit.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        # Show available presets
        self.console.print(f"\n[bold]Available Presets:[/bold]")
        for i, preset_name in enumerate(presets, 1):
            current_marker = " [green]← current[/green]" if preset_name == config.current_preset else ""
            self.console.print(f"  [cyan]{i}.[/cyan] {preset_name.title()}{current_marker}")
        
        self.console.print(f"\n[dim]Select preset to edit (1-{len(presets)}) or q to cancel:[/dim]")
        self.console.print("\n[cyan bold]EDIT>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if choice is None or choice.lower() == 'q':
            return False
        
        try:
            preset_num = int(choice)
            if 1 <= preset_num <= len(presets):
                preset_name = presets[preset_num - 1]
                return self.edit_single_preset(llm_client, preset_name)
            else:
                self.console.print(f"[red]Invalid selection. Choose 1-{len(presets)}[/red]")
                return False
        except ValueError:
            self.console.print("[red]Invalid input. Enter a number or 'q'.[/red]")
            return False

    def edit_single_preset(self, llm_client: 'LLMClient', preset_name: str) -> bool:
        """Edit a specific preset's name or configuration."""
        config = llm_client.config
        preset_config = config.presets[preset_name]
        changes_made = False
        
        while True:
            self.console.print(f"\n[bold cyan]✏️  Editing Preset: {preset_name.title()}[/bold cyan]")
            
            # Show current configuration
            current_config = f"\n[bold]Current Configuration:[/bold]\n"
            for workflow_name, workflow_config in preset_config.items():
                fallback_text = ""
                if workflow_config.fallback:
                    fallback_text = f" (fallback: {workflow_config.fallback['provider']}/{workflow_config.fallback['model']})"
                current_config += f"  {workflow_name.replace('_', ' ').title()}: [yellow]{workflow_config.provider}/{workflow_config.model}[/yellow]{fallback_text}\n"
            
            self.console.print(current_config)
            
            # Show edit options
            edit_options = f"""\n[bold]Edit Options:[/bold]
  [green]1.[/green] Rename preset
  [green]2.[/green] Edit workflow configurations
  [green]3.[/green] Duplicate preset with new name
  [green]t.[/green] Test preset configuration
  [green]s.[/green] Save changes
  [green]q.[/green] Cancel (discard changes)"""
            
            self.console.print(edit_options)
            self.console.print("\n[cyan bold]EDIT>[/cyan bold] ", end="")
            edit_choice = self.safe_input()
            
            if edit_choice is None or edit_choice.lower() == 'q':
                if changes_made:
                    self.console.print("[yellow]Changes discarded.[/yellow]")
                return False
            edit_choice = edit_choice.lower()
            
            if edit_choice == '1':
                # Rename preset
                self.console.print(f"\n[dim]Current name: {preset_name}[/dim]")
                self.console.print("[dim]Enter new preset name:[/dim]")
                self.console.print("[green]New name>[/green] ", end="")
                new_name = self.safe_input()
                
                if new_name and new_name.strip():
                    new_name = new_name.strip()
                    if new_name != preset_name:
                        if new_name not in config.presets:
                            # Rename the preset
                            config.presets[new_name] = preset_config
                            del config.presets[preset_name]
                            
                            # Update current preset if this was the active one
                            if config.current_preset == preset_name:
                                config.current_preset = new_name
                            
                            self.console.print(f"[green]✅ Preset renamed from '{preset_name}' to '{new_name}'[/green]")
                            preset_name = new_name  # Update local variable
                            changes_made = True
                        else:
                            self.console.print(f"[red]Preset '{new_name}' already exists[/red]")
                            
            elif edit_choice == '2':
                # Edit workflow configurations (reuse workflow configuration logic)
                self.console.print(f"\n[bold yellow]🔧 Configure Workflows for {preset_name}[/bold yellow]")
                
                workflows = ["processing", "conversation", "jira_matching"]
                workflow_descriptions = {
                    "processing": "Extracts structured data from journal entries",
                    "conversation": "Handles interactive refinement and questions", 
                    "jira_matching": "Matches work to JIRA tickets"
                }
                
                # Show current workflow assignments for this preset
                self.console.print(f"\n[bold]Current Workflow Assignments:[/bold]")
                for i, workflow in enumerate(workflows, 1):
                    current = preset_config[workflow]
                    fallback_text = ""
                    if current.fallback:
                        fallback_text = f" (fallback: {current.fallback['provider']}/{current.fallback['model']})"
                    
                    self.console.print(f"  [cyan]{i}.[/cyan] [bold]{workflow.replace('_', ' ').title()}[/bold]")
                    self.console.print(f"      {workflow_descriptions[workflow]}")
                    self.console.print(f"      [green]→ {current.provider}/{current.model}[/green]{fallback_text}")
                
                self.console.print(f"\n[dim]Select workflow to configure (1-{len(workflows)}) or Enter to go back:[/dim]")
                self.console.print("\n[cyan bold]WORKFLOW>[/cyan bold] ", end="")
                workflow_choice = self.safe_input()
                
                if workflow_choice and workflow_choice.strip():
                    try:
                        workflow_num = int(workflow_choice)
                        if 1 <= workflow_num <= len(workflows):
                            workflow_name = workflows[workflow_num - 1]
                            if self.configure_preset_workflow(llm_client, preset_config, workflow_name):
                                changes_made = True
                        else:
                            self.console.print(f"[red]Invalid selection. Choose 1-{len(workflows)}[/red]")
                    except ValueError:
                        self.console.print("[red]Invalid input. Enter a number or press Enter to go back.[/red]")
                        
            elif edit_choice == '3':
                # Duplicate preset
                self.console.print(f"\n[bold green]📄 Duplicate Preset[/bold green]")
                self.console.print(f"[dim]Source: {preset_name}[/dim]")
                self.console.print("[dim]Enter name for the duplicate:[/dim]")
                self.console.print("[green]Duplicate name>[/green] ", end="")
                duplicate_name = self.safe_input()
                
                if duplicate_name and duplicate_name.strip():
                    duplicate_name = duplicate_name.strip()
                    if duplicate_name not in config.presets:
                        # Create a deep copy of the preset
                        config.presets[duplicate_name] = {
                            workflow_name: workflow_config.model_copy()
                            for workflow_name, workflow_config in preset_config.items()
                        }
                        self.console.print(f"[green]✅ Preset duplicated as '{duplicate_name}'[/green]")
                        changes_made = True
                    else:
                        self.console.print(f"[red]Preset '{duplicate_name}' already exists[/red]")
                        
            elif edit_choice == 't':
                # Test preset configuration
                self.console.print(f"\n[dim]Testing preset '{preset_name}' configuration...[/dim]")
                
                # Temporarily apply this preset for testing
                original_workflows = config.workflows.copy()
                original_preset = config.current_preset
                
                try:
                    config.workflows = {
                        workflow_name: workflow_config
                        for workflow_name, workflow_config in preset_config.items()
                    }
                    config.current_preset = preset_name
                    
                    success = self.test_model_connection(llm_client, silent=True)
                    
                    if success:
                        self.console.print("[green]✅ Preset configuration tests passed![/green]")
                    else:
                        self.console.print("[yellow]⚠️  Some configurations in this preset have issues[/yellow]")
                        
                finally:
                    # Restore original configuration
                    config.workflows = original_workflows
                    config.current_preset = original_preset
                    
            elif edit_choice == 's':
                # Save changes
                if changes_made:
                    llm_client.storage.save_config(config)
                    self.console.print(f"[green]✅ Preset '{preset_name}' saved successfully[/green]")
                    return True
                else:
                    self.console.print("[yellow]No changes to save[/yellow]")
                    return False
                    
            elif edit_choice == '':
                # Empty command, just redraw
                pass
            else:
                self.console.print(f"[red]Unknown command: '{edit_choice}'. Use options from the menu.[/red]")

    def configure_preset_workflow(self, llm_client: 'LLMClient', preset_config, workflow_name: str) -> bool:
        """Configure a single workflow within a preset."""
        self.console.print(f"\n[bold green]🎯 Configure {workflow_name.replace('_', ' ').title()}[/bold green]")
        
        config = llm_client.config
        current_setting = preset_config[workflow_name]
        
        # Show current setting
        self.console.print(f"Current: [yellow]{current_setting.provider}/{current_setting.model}[/yellow]")
        
        # Get provider choice
        providers = list(config.providers.keys())
        self.console.print(f"\n[bold]Available Providers:[/bold]")
        for i, provider in enumerate(providers, 1):
            status = "●" if provider == current_setting.provider else "○"
            self.console.print(f"  [cyan]{i}.[/cyan] {status} {provider.title()}")
        
        self.console.print(f"\n[dim]Select provider (1-{len(providers)}) or Enter to keep current:[/dim]")
        self.console.print("[green]Provider>[/green] ", end="")
        provider_choice = self.safe_input()
        if provider_choice is None:
            return False
        
        # Parse provider choice
        if not provider_choice:
            # Keep current provider
            selected_provider = current_setting.provider
        else:
            try:
                provider_num = int(provider_choice)
                if 1 <= provider_num <= len(providers):
                    selected_provider = providers[provider_num - 1]
                else:
                    self.console.print(f"[red]Invalid provider selection[/red]")
                    return False
            except ValueError:
                self.console.print(f"[red]Invalid input[/red]")
                return False
        
        # Get available models for selected provider
        self.console.print(f"\n[dim]🤖⏳ Fetching models for {selected_provider}...[/dim]")
        available_models = llm_client.get_available_models(selected_provider)
        
        if not available_models:
            self.console.print(f"[red]❌ No models available for {selected_provider}[/red]")
            return False
        
        # Show model choices
        self.console.print(f"\n[bold]Available Models for {selected_provider.title()}:[/bold]")
        for i, model in enumerate(available_models, 1):
            status = "●" if model == current_setting.model and selected_provider == current_setting.provider else "○"
            self.console.print(f"  [cyan]{i}.[/cyan] {status} {model}")
        
        self.console.print(f"\n[dim]Select model (1-{len(available_models)}) or Enter to keep current:[/dim]")
        self.console.print("[green]Model>[/green] ", end="")
        model_choice = self.safe_input()
        if model_choice is None:
            return False
        
        # Parse model choice
        if not model_choice:
            # Keep current model if same provider, otherwise use first available
            if selected_provider == current_setting.provider:
                selected_model = current_setting.model
            else:
                selected_model = available_models[0]
        else:
            try:
                model_num = int(model_choice)
                if 1 <= model_num <= len(available_models):
                    selected_model = available_models[model_num - 1]
                else:
                    self.console.print(f"[red]Invalid model selection[/red]")
                    return False
            except ValueError:
                self.console.print(f"[red]Invalid input[/red]")
                return False
        
        # Update preset workflow configuration
        from .models import WorkflowSettings
        preset_config[workflow_name] = WorkflowSettings(
            provider=selected_provider,
            model=selected_model,
            fallback=current_setting.fallback  # Keep existing fallback
        )
        
        self.console.print(f"[green]✅ {workflow_name.replace('_', ' ').title()} updated: {selected_provider}/{selected_model}[/green]")
        return True

    def manage_providers(self, llm_client: 'LLMClient') -> bool:
        """Manage provider configurations."""
        self.console.print(f"\n[bold blue]🔧 Manage Providers[/bold blue]")
        
        config = llm_client.config
        changes_made = False
        
        while True:
            # Show current providers 
            self.console.print(f"\n[bold]Current Providers:[/bold]")
            providers = list(config.providers.keys())
            
            if not providers:
                self.console.print("  [dim]No providers configured yet.[/dim]")
                self.console.print("\n[yellow]💡 Tip:[/yellow] You can load default provider templates to get started quickly!")
                # Show options for empty state
                self.console.print(f"\n[dim]Options: [yellow]l[/yellow]=load templates, [yellow]a[/yellow]=add new, [yellow]q[/yellow]=back[/dim]")
            else:
                for i, provider_name in enumerate(providers, 1):
                    provider_config = config.providers[provider_name]
                    auth_info = "API Key" if provider_config.api_key else f"ENV: {provider_config.auth_env}" if provider_config.auth_env else "No Auth"
                    self.console.print(f"  [cyan]{i}.[/cyan] [bold]{provider_config.service_name}[/bold] ({provider_config.protocol})")
                    self.console.print(f"      Endpoint: {provider_config.api_base}")
                    self.console.print(f"      Auth: {auth_info}")
                
                # Show options for populated state
                self.console.print(f"\n[dim]Options: 1-{len(providers)} to edit provider, [yellow]l[/yellow]=load templates, [yellow]a[/yellow]=add new, [yellow]d[/yellow]=delete, [yellow]t[/yellow]=test all, [yellow]e[/yellow]=check env, [yellow]s[/yellow]=save, [yellow]q[/yellow]=back[/dim]")
            
            self.console.print("\n[cyan bold]PROVIDER>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            choice = choice.lower()
            
            if choice == 'a':
                if self.add_provider(llm_client):
                    changes_made = True
            elif choice == 'l':
                if self._load_provider_templates(llm_client):
                    changes_made = True
            elif choice == 'd':
                if self.delete_provider(llm_client):
                    changes_made = True
            elif choice == 't':
                self.test_all_providers(llm_client)
            elif choice == 'e':
                self._check_environment_variables(llm_client)
            elif choice == 's':
                if changes_made:
                    llm_client.storage.save_config(config)
                    self.console.print("[green]✅ Provider configuration saved[/green]")
                    return True
                else:
                    self.console.print("[yellow]No changes to save[/yellow]")
            else:
                # Try to parse as provider number
                try:
                    provider_num = int(choice)
                    if 1 <= provider_num <= len(providers):
                        provider_name = providers[provider_num - 1]
                        if self.edit_provider(llm_client, provider_name):
                            changes_made = True
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(providers)}[/red]")
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number or letter option.[/red]")
        
        return changes_made
    
    def _load_provider_templates(self, llm_client: 'LLMClient') -> bool:
        """Load default provider templates from JSON file."""
        self.console.clear()
        self.console.print("\n[bold green]📋 Load Provider Templates[/bold green]")
        
        # Copy template file to user directory
        template_path = llm_client.storage.copy_default_providers_template()
        
        if not template_path.exists():
            self.console.print("[red]❌ Default providers template not found![/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        try:
            # Load template file
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            providers_data = template_data.get('providers', {})
            
            self.console.print(f"[dim]{template_data.get('description', 'Default provider templates')}[/dim]")
            self.console.print(f"\n[bold]Available Templates:[/bold]")
            
            # Show available templates
            for i, (name, provider_info) in enumerate(providers_data.items(), 1):
                desc = provider_info.get('description', 'No description')
                self.console.print(f"  [cyan]{i}.[/cyan] [bold]{name.title()}[/bold] - {desc}")
            
            self.console.print(f"\n[yellow]Options:[/yellow]")
            self.console.print(f"  [green]all[/green] - Load all templates")
            self.console.print(f"  [green]1-{len(providers_data)}[/green] - Load specific template")
            self.console.print(f"  [green]q[/green] - Cancel")
            
            self.console.print(f"\n[cyan bold]Load which templates>[/cyan bold] ", end="")
            choice = self.safe_input()
            
            if choice is None or choice.lower() == 'q':
                return False
            
            from .models import ProviderConfig
            config = llm_client.config
            loaded_count = 0
            
            if choice.lower() == 'all':
                # Load all templates
                for name, provider_info in providers_data.items():
                    provider_config = ProviderConfig(
                        api_base=provider_info['api_base'],
                        protocol=provider_info['protocol'],
                        service_name=provider_info['service_name'],
                        api_key=provider_info.get('api_key'),
                        auth_env=provider_info.get('auth_env')
                    )
                    config.providers[name] = provider_config
                    loaded_count += 1
                    
                self.console.print(f"\n[green]✅ Loaded {loaded_count} provider templates[/green]")
            else:
                # Load specific template
                try:
                    template_num = int(choice)
                    provider_names = list(providers_data.keys())
                    if 1 <= template_num <= len(provider_names):
                        name = provider_names[template_num - 1]
                        provider_info = providers_data[name]
                        
                        provider_config = ProviderConfig(
                            api_base=provider_info['api_base'],
                            protocol=provider_info['protocol'],
                            service_name=provider_info['service_name'],
                            api_key=provider_info.get('api_key'),
                            auth_env=provider_info.get('auth_env')
                        )
                        config.providers[name] = provider_config
                        loaded_count = 1
                        
                        self.console.print(f"\n[green]✅ Loaded {name} provider template[/green]")
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(provider_names)}[/red]")
                        self.console.print("\n[dim]Press Enter to continue...[/dim]")
                        self.safe_input()
                        return False
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number or 'all'[/red]")
                    self.console.print("\n[dim]Press Enter to continue...[/dim]")
                    self.safe_input()
                    return False
            
            if loaded_count > 0:
                self.console.print(f"\n[yellow]⚠️  Important:[/yellow] These are templates with default settings.")
                self.console.print("You may need to:")
                self.console.print("  • Set up API keys for cloud providers")
                self.console.print("  • Verify local server endpoints are correct")
                self.console.print("  • Test connections before using")
                
                self.console.print("\n[dim]Press Enter to continue...[/dim]")
                self.safe_input()
                return True
            
            return False
            
        except Exception as e:
            self.console.print(f"[red]❌ Error loading templates: {e}[/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False

    def add_provider(self, llm_client: 'LLMClient') -> bool:
        """Add a new provider configuration."""
        self.console.print(f"\n[bold cyan]➕ Add New Provider[/bold cyan]")
        
        # Get provider name
        self.console.print("[dim]Enter provider name (e.g., 'custom-openai', 'local-llama'):[/dim]")
        self.console.print("[green]Provider name>[/green] ", end="")
        provider_name = self.safe_input()
        if provider_name is None or not provider_name.strip():
            return False
        
        provider_name = provider_name.strip().lower()
        
        # Check if already exists
        if provider_name in llm_client.config.providers:
            self.console.print(f"[red]Provider '{provider_name}' already exists[/red]")
            return False
        
        # Get API endpoint
        self.console.print("[dim]Enter API endpoint URL:[/dim]")
        self.console.print("[green]API endpoint>[/green] ", end="")
        api_base = self.safe_input()
        if api_base is None or not api_base.strip():
            return False
        
        api_base = api_base.strip()
        
        # Get authentication method
        self.console.print("\n[bold]Authentication Method:[/bold]")
        self.console.print("  [cyan]1.[/cyan] API Key (direct)")
        self.console.print("  [cyan]2.[/cyan] Environment variable")
        self.console.print("  [cyan]3.[/cyan] No authentication")
        
        self.console.print("\n[green]Auth method (1-3)>[/green] ", end="")
        auth_choice = self.safe_input()
        if auth_choice is None:
            return False
        
        api_key = None
        auth_env = None
        
        if auth_choice == '1':
            self.console.print("[dim]Enter API key:[/dim]")
            self.console.print("[green]API key>[/green] ", end="")
            api_key = self.safe_input()
            if api_key is None:
                return False
        elif auth_choice == '2':
            self.console.print("[dim]Enter environment variable name (e.g., 'MY_API_KEY'):[/dim]")
            self.console.print("[green]Env var>[/green] ", end="")
            auth_env = self.safe_input()
            if auth_env is None:
                return False
        elif auth_choice != '3':
            self.console.print("[red]Invalid choice[/red]")
            return False
        
        # Create provider config
        from .models import ProviderConfig
        llm_client.config.providers[provider_name] = ProviderConfig(
            api_base=api_base,
            api_key=api_key if api_key else None,
            auth_env=auth_env if auth_env else None
        )
        
        self.console.print(f"[green]✅ Provider '{provider_name}' added successfully[/green]")
        return True

    def show_config_help(self):
        """Show help for configuration options."""
        help_text = """
[bold]Configuration Help:[/bold]

[yellow]1. Configure Workflows[/yellow]
   Assign different AI models to different tasks:
   • [green]Processing[/green]: Extracts structured data from your journal entries
   • [green]Conversation[/green]: Handles interactive chat and refinement
   • [green]JIRA Matching[/green]: Matches your work to ticket systems

[yellow]2. Manage Providers[/yellow]
   Add or edit API endpoints for different AI services:
   • Local providers: LM Studio, Ollama
   • Cloud providers: OpenAI, Anthropic, etc.
   • Custom endpoints: Self-hosted or enterprise APIs

[yellow]3. Create Presets[/yellow]
   Save your current configuration as a named preset:
   • Quick switching between different setups
   • Share configurations between team members
   • Backup your optimal configurations

[bold]Best Practices:[/bold]
• Use fast local models for conversation, powerful cloud models for processing
• Test configurations before saving as presets
• Keep fallback models configured for reliability
        """
        
        self.console.print(Panel(help_text.strip(), title="Configuration Help", border_style="blue"))

    def setup_templates(self, llm_client: 'LLMClient') -> bool:
        """Quick setup using configuration templates."""
        self.console.print(f"\n[bold magenta]🚀 Quick Setup Templates[/bold magenta]")
        self.console.print("Choose a template that matches your setup:")
        
        templates = {
            "1": {
                "name": "Local Only",
                "description": "All workflows use local models (LM Studio + Ollama)",
                "use_case": "Privacy-focused, no internet required"
            },
            "2": {
                "name": "Hybrid",
                "description": "Local for conversation, cloud for processing", 
                "use_case": "Balance between privacy and performance"
            },
            "3": {
                "name": "Cloud Optimized",
                "description": "Best cloud models for all workflows",
                "use_case": "Maximum performance and capabilities"
            },
            "4": {
                "name": "Cost Effective",
                "description": "Cheaper models optimized for cost",
                "use_case": "Budget-conscious with good performance"
            }
        }
        
        while True:
            self.console.print(f"\n[bold]Available Templates:[/bold]")
            for key, template in templates.items():
                self.console.print(f"  [cyan]{key}.[/cyan] [bold]{template['name']}[/bold]")
                self.console.print(f"      {template['description']}")
                self.console.print(f"      [dim]Best for: {template['use_case']}[/dim]")
            
            self.console.print(f"\n[dim]Choose template (1-{len(templates)}) or q to cancel:[/dim]")
            self.console.print("\n[cyan bold]TEMPLATE>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                return False
            
            if choice in templates:
                return self.apply_template(llm_client, choice, templates[choice]['name'])
            else:
                self.console.print(f"[red]Invalid choice. Choose 1-{len(templates)} or 'q'[/red]")

    def apply_template(self, llm_client: 'LLMClient', template_id: str, template_name: str) -> bool:
        """Apply a specific configuration template."""
        self.console.print(f"\n[bold yellow]⚙️  Applying {template_name} Template[/bold yellow]")
        
        from .models import WorkflowSettings
        config = llm_client.config
        
        # Define template configurations
        if template_id == "1":  # Local Only
            workflows = {
                "processing": WorkflowSettings(provider="lmstudio", model="llama-3.2-3b-instruct"),
                "conversation": WorkflowSettings(provider="ollama", model="llama3.2:3b"),
                "jira_matching": WorkflowSettings(provider="lmstudio", model="llama-3.2-3b-instruct")
            }
        elif template_id == "2":  # Hybrid
            workflows = {
                "processing": WorkflowSettings(provider="openai", model="gpt-4o-mini", 
                                             fallback={"provider": "lmstudio", "model": "llama-3.2-3b-instruct"}),
                "conversation": WorkflowSettings(provider="lmstudio", model="llama-3.2-3b-instruct"),
                "jira_matching": WorkflowSettings(provider="openai", model="gpt-4o-mini",
                                                fallback={"provider": "lmstudio", "model": "llama-3.2-3b-instruct"})
            }
        elif template_id == "3":  # Cloud Optimized  
            workflows = {
                "processing": WorkflowSettings(provider="openai", model="gpt-4o",
                                             fallback={"provider": "openai", "model": "gpt-4o-mini"}),
                "conversation": WorkflowSettings(provider="anthropic", model="claude-3-5-sonnet-20241022",
                                               fallback={"provider": "openai", "model": "gpt-4o-mini"}),
                "jira_matching": WorkflowSettings(provider="openai", model="gpt-4o-mini",
                                                fallback={"provider": "openai", "model": "gpt-3.5-turbo"})
            }
        elif template_id == "4":  # Cost Effective
            workflows = {
                "processing": WorkflowSettings(provider="openai", model="gpt-4o-mini",
                                             fallback={"provider": "lmstudio", "model": "llama-3.2-3b-instruct"}),
                "conversation": WorkflowSettings(provider="openai", model="gpt-3.5-turbo",
                                               fallback={"provider": "lmstudio", "model": "llama-3.2-3b-instruct"}),
                "jira_matching": WorkflowSettings(provider="openai", model="gpt-3.5-turbo",
                                                fallback={"provider": "lmstudio", "model": "llama-3.2-3b-instruct"})
            }
        else:
            return False
        
        # Apply the template
        config.workflows = workflows
        
        # Create a preset with this template
        preset_name = f"{template_name.lower().replace(' ', '_')}_template"
        config.presets[preset_name] = {
            workflow_name: workflow_config.model_copy()
            for workflow_name, workflow_config in workflows.items()
        }
        config.current_preset = preset_name
        
        # Save configuration
        llm_client.storage.save_config(config)
        
        self.console.print(f"[green]✅ {template_name} template applied successfully![/green]")
        self.console.print(f"[green]✅ Created preset: '{preset_name}'[/green]")
        
        # Test the configuration
        self.console.print(f"\n[dim]Testing the new configuration...[/dim]")
        if self.validate_configuration(llm_client, silent=True):
            self.console.print("[green]✅ Template configuration is valid[/green]")
        else:
            self.console.print("[yellow]⚠️  Template has some issues - you may need to adjust providers[/yellow]")
        
        self.console.print(f"\n[bold]What's Next?[/bold]")
        if template_id in ["2", "3", "4"]:  # Templates using cloud providers
            self.console.print("• Make sure your API keys are configured (OpenAI_API_KEY, ANTHROPIC_API_KEY)")
            self.console.print("• Test connections using 't' in the model settings")
        if template_id in ["1", "2"]:  # Templates using local providers
            self.console.print("• Make sure LM Studio is running with a model loaded")
            self.console.print("• Make sure Ollama is running with models installed")
        
        self.console.print(f"\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
        
        return True

    def import_export_config(self, llm_client: 'LLMClient'):
        """Import or export configuration."""
        self.console.clear()
        self.console.print(f"\n[bold magenta]📁 Import/Export Configuration[/bold magenta]")
        self.console.print("\n[dim]Feature coming soon! Will support JSON import/export of configurations.[/dim]")
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()

    def edit_provider(self, llm_client: 'LLMClient', provider_name: str) -> bool:
        """Edit an existing provider configuration."""
        self.console.print(f"\n[bold yellow]✏️  Edit Provider: {provider_name.title()}[/bold yellow]")
        
        config = llm_client.config
        provider_config = config.providers[provider_name]
        changes_made = False
        
        while True:
            # Show current configuration
            auth_info = "API Key (set)" if provider_config.api_key else f"ENV: {provider_config.auth_env}" if provider_config.auth_env else "No Auth"
            
            current_config = f"""
[bold]Current Configuration:[/bold]
  Provider Name: {provider_name}
  API Endpoint: {provider_config.api_base}
  Authentication: {auth_info}
            """
            
            self.console.print(current_config)
            
            # Show edit options
            edit_options = """
[bold]Edit Options:[/bold]
  [green]1.[/green] Change API endpoint
  [green]2.[/green] Update authentication
  [green]3.[/green] Rename provider
  [green]t.[/green] Test connection
  [green]s.[/green] Save changes
  [green]q.[/green] Cancel (discard changes)
            """
            
            self.console.print(edit_options)
            self.console.print("\n[yellow bold]EDIT>[/yellow bold] ", end="")
            choice = self.safe_input()
            
            if choice is None or choice.lower() == 'q':
                if changes_made:
                    self.console.print("[yellow]Changes discarded.[/yellow]")
                return False
            choice = choice.lower()
            
            if choice == '1':
                # Edit API endpoint
                self.console.print(f"\n[dim]Current endpoint: {provider_config.api_base}[/dim]")
                self.console.print("[dim]Enter new API endpoint (or Enter to keep current):[/dim]")
                self.console.print("[green]New endpoint>[/green] ", end="")
                new_endpoint = self.safe_input()
                
                if new_endpoint and new_endpoint.strip():
                    new_endpoint = new_endpoint.strip()
                    if new_endpoint.startswith(('http://', 'https://')):
                        provider_config.api_base = new_endpoint
                        self.console.print(f"[green]✅ Endpoint updated to: {new_endpoint}[/green]")
                        changes_made = True
                    else:
                        self.console.print("[red]Invalid URL - must start with http:// or https://[/red]")
                        
            elif choice == '2':
                # Edit authentication
                current_auth = "API Key" if provider_config.api_key else f"Environment variable ({provider_config.auth_env})" if provider_config.auth_env else "None"
                self.console.print(f"\n[dim]Current authentication: {current_auth}[/dim]")
                self.console.print("\n[bold]New Authentication Method:[/bold]")
                self.console.print("  [cyan]1.[/cyan] API Key (direct)")
                self.console.print("  [cyan]2.[/cyan] Environment variable")
                self.console.print("  [cyan]3.[/cyan] No authentication")
                self.console.print("  [cyan]Enter.[/cyan] Keep current")
                
                self.console.print("\n[green]Auth method>[/green] ", end="")
                auth_choice = self.safe_input()
                
                if auth_choice == '1':
                    self.console.print("[dim]Enter API key:[/dim]")
                    self.console.print("[green]API key>[/green] ", end="")
                    api_key = self.safe_input()
                    if api_key:
                        provider_config.api_key = api_key
                        provider_config.auth_env = None
                        self.console.print("[green]✅ API key updated[/green]")
                        changes_made = True
                elif auth_choice == '2':
                    self.console.print("[dim]Enter environment variable name (e.g., 'MY_API_KEY'):[/dim]")
                    self.console.print("[green]Env var>[/green] ", end="")
                    auth_env = self.safe_input()
                    if auth_env:
                        provider_config.auth_env = auth_env
                        provider_config.api_key = None
                        self.console.print(f"[green]✅ Authentication set to environment variable: {auth_env}[/green]")
                        changes_made = True
                elif auth_choice == '3':
                    provider_config.api_key = None
                    provider_config.auth_env = None
                    self.console.print("[green]✅ Authentication disabled[/green]")
                    changes_made = True
                    
            elif choice == '3':
                # Rename provider
                self.console.print(f"\n[dim]Current name: {provider_name}[/dim]")
                self.console.print("[dim]Enter new provider name:[/dim]")
                self.console.print("[green]New name>[/green] ", end="")
                new_name = self.safe_input()
                
                if new_name and new_name.strip():
                    new_name = new_name.strip().lower()
                    if new_name != provider_name:
                        if new_name not in config.providers:
                            # Rename the provider
                            config.providers[new_name] = provider_config
                            del config.providers[provider_name]
                            
                            # Update all references in workflows
                            for workflow in config.workflows.values():
                                if workflow.provider == provider_name:
                                    workflow.provider = new_name
                                if workflow.fallback and workflow.fallback.get('provider') == provider_name:
                                    workflow.fallback['provider'] = new_name
                            
                            # Update all references in presets
                            for preset_workflows in config.presets.values():
                                for workflow in preset_workflows.values():
                                    if workflow.provider == provider_name:
                                        workflow.provider = new_name
                                    if workflow.fallback and workflow.fallback.get('provider') == provider_name:
                                        workflow.fallback['provider'] = new_name
                            
                            self.console.print(f"[green]✅ Provider renamed from '{provider_name}' to '{new_name}'[/green]")
                            provider_name = new_name  # Update local variable
                            changes_made = True
                        else:
                            self.console.print(f"[red]Provider '{new_name}' already exists[/red]")
                            
            elif choice == 't':
                # Test connection
                self.console.print(f"\n[dim]Testing connection to {provider_name}...[/dim]")
                models = llm_client.get_available_models(provider_name)
                if models:
                    self.console.print(f"[green]✅ Connection successful - {len(models)} models available[/green]")
                else:
                    self.console.print(f"[red]❌ Connection failed[/red]")
                    self.show_connection_help(provider_name, "test", llm_client)
                    
            elif choice == 's':
                # Save changes
                if changes_made:
                    llm_client.storage.save_config(config)
                    self.console.print(f"[green]✅ Provider '{provider_name}' saved successfully[/green]")
                    return True
                else:
                    self.console.print("[yellow]No changes to save[/yellow]")
                    return False
                    
            elif choice == '':
                # Empty command, just redraw
                pass
            else:
                self.console.print(f"[red]Unknown command: '{choice}'. Use options from the menu.[/red]")

    def delete_provider(self, llm_client: 'LLMClient') -> bool:
        """Delete a provider configuration."""
        self.console.print(f"\n[bold red]🗑️  Delete Provider[/bold red]")
        
        config = llm_client.config
        providers = list(config.providers.keys())
        
        if not providers:
            self.console.print("[yellow]No providers to delete.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return False
        
        # Show available providers
        self.console.print(f"\n[bold red]⚠️  WARNING: Deleting a provider will break any workflows using it![/bold red]")
        self.console.print(f"\n[bold]Available Providers:[/bold]")
        for i, provider_name in enumerate(providers, 1):
            # Check if provider is in use
            in_use = []
            for workflow_name, workflow in config.workflows.items():
                if workflow.provider == provider_name:
                    in_use.append(workflow_name)
                if workflow.fallback and workflow.fallback.get('provider') == provider_name:
                    in_use.append(f"{workflow_name} (fallback)")
            
            usage_text = ""
            if in_use:
                usage_text = f" [red]← USED BY: {', '.join(in_use)}[/red]"
            
            self.console.print(f"  [cyan]{i}.[/cyan] {provider_name.title()}{usage_text}")
        
        self.console.print(f"\n[dim]Select provider to delete (1-{len(providers)}) or q to cancel:[/dim]")
        self.console.print("\n[red bold]DELETE>[/red bold] ", end="")
        choice = self.safe_input()
        
        if choice is None or choice.lower() == 'q':
            return False
        
        try:
            provider_num = int(choice)
            if 1 <= provider_num <= len(providers):
                provider_name = providers[provider_num - 1]
                return self.confirm_provider_deletion(llm_client, provider_name)
            else:
                self.console.print(f"[red]Invalid selection. Choose 1-{len(providers)}[/red]")
                return False
        except ValueError:
            self.console.print("[red]Invalid input. Enter a number or 'q'.[/red]")
            return False

    def confirm_provider_deletion(self, llm_client: 'LLMClient', provider_name: str) -> bool:
        """Confirm provider deletion with safety checks."""
        config = llm_client.config
        
        # Check what would be affected
        affected_workflows = []
        affected_presets = []
        
        # Check current workflows
        for workflow_name, workflow in config.workflows.items():
            if workflow.provider == provider_name:
                affected_workflows.append(f"{workflow_name} (primary)")
            if workflow.fallback and workflow.fallback.get('provider') == provider_name:
                affected_workflows.append(f"{workflow_name} (fallback)")
        
        # Check presets
        for preset_name, preset_workflows in config.presets.items():
            for workflow_name, workflow in preset_workflows.items():
                if workflow.provider == provider_name:
                    affected_presets.append(f"{preset_name}::{workflow_name}")
                if workflow.fallback and workflow.fallback.get('provider') == provider_name:
                    affected_presets.append(f"{preset_name}::{workflow_name} (fallback)")
        
        # Show impact analysis
        self.console.print(f"\n[bold red]🚨 Impact Analysis: Deleting '{provider_name}'[/bold red]")
        
        if affected_workflows:
            self.console.print(f"\n[red]Will break current workflows:[/red]")
            for workflow in affected_workflows:
                self.console.print(f"  • {workflow}")
        
        if affected_presets:
            self.console.print(f"\n[red]Will break presets:[/red]")
            for preset in affected_presets:
                self.console.print(f"  • {preset}")
        
        if not affected_workflows and not affected_presets:
            self.console.print(f"\n[green]✅ Safe to delete - no workflows or presets are using this provider[/green]")
        
        # Final confirmation
        self.console.print(f"\n[bold red]This action cannot be undone![/bold red]")
        self.console.print(f"[red bold]Type 'DELETE {provider_name.upper()}' to confirm>[/red bold] ", end="")
        confirmation = self.safe_input()
        
        if confirmation == f"DELETE {provider_name.upper()}":
            # Remove provider
            del config.providers[provider_name]
            
            # Clean up broken references (optional - could also just warn)
            self.console.print(f"\n[yellow]Cleaning up broken references...[/yellow]")
            
            # Fix workflows by removing broken ones or switching to available providers
            for workflow_name, workflow in list(config.workflows.items()):
                if workflow.provider == provider_name:
                    # Try to use fallback if available
                    if workflow.fallback and workflow.fallback['provider'] in config.providers:
                        config.workflows[workflow_name].provider = workflow.fallback['provider']
                        config.workflows[workflow_name].model = workflow.fallback['model']
                        config.workflows[workflow_name].fallback = None
                        self.console.print(f"  • {workflow_name}: Switched to fallback {workflow.fallback['provider']}/{workflow.fallback['model']}")
                    else:
                        # Use first available provider as emergency fallback
                        if config.providers:
                            emergency_provider = list(config.providers.keys())[0]
                            config.workflows[workflow_name].provider = emergency_provider
                            config.workflows[workflow_name].model = "default"  # Will need manual fix
                            self.console.print(f"  • {workflow_name}: Emergency switch to {emergency_provider} (requires manual model selection)")
                
                if workflow.fallback and workflow.fallback.get('provider') == provider_name:
                    config.workflows[workflow_name].fallback = None
                    self.console.print(f"  • {workflow_name}: Removed broken fallback")
            
            # Save configuration
            llm_client.storage.save_config(config)
            
            self.console.print(f"\n[green]✅ Provider '{provider_name}' deleted successfully[/green]")
            if affected_workflows or affected_presets:
                self.console.print(f"[yellow]⚠️  Please review and fix your workflow configurations[/yellow]")
            
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return True
        else:
            self.console.print(f"[yellow]Deletion cancelled - confirmation did not match[/yellow]")
            return False

    def test_all_providers(self, llm_client: 'LLMClient'):
        """Test connection to all configured providers."""
        self.console.print(f"\n[bold blue]🔍 Test All Providers[/bold blue]")
        
        # Reload environment variables in case they were added during this session
        llm_client.reload_env()
        
        config = llm_client.config
        for provider_name in config.providers.keys():
            self.console.print(f"\n[dim]Testing {provider_name}...[/dim]")
            try:
                models = llm_client.get_available_models(provider_name)
                if models:
                    self.console.print(f"[green]✅ {provider_name}: {len(models)} models available[/green]")
                else:
                    self.console.print(f"[red]❌ {provider_name}: No models found or connection failed[/red]")
            except Exception as e:
                self.console.print(f"[red]❌ {provider_name}: {str(e)}[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _check_environment_variables(self, llm_client: 'LLMClient'):
        """Check environment variables for debugging."""
        self.console.clear()
        self.console.print("\n[bold blue]🔍 Environment Variables Check[/bold blue]")
        
        # Reload environment first
        llm_client.reload_env()
        
        config = llm_client.config
        
        # Show .env file locations
        from pathlib import Path
        local_env = Path(".env")
        global_env = llm_client.storage.base_path / ".env"
        
        self.console.print(f"\n[bold]Environment Files:[/bold]")
        self.console.print(f"  Local .env: {local_env.absolute()} {'✅' if local_env.exists() else '❌'}")
        self.console.print(f"  Global .env: {global_env} {'✅' if global_env.exists() else '❌'}")
        
        # Check environment variables for each provider
        self.console.print(f"\n[bold]Environment Variables for Providers:[/bold]")
        for provider_name, provider_config in config.providers.items():
            self.console.print(f"\n  [cyan]{provider_name.title()}:[/cyan]")
            if provider_config.auth_env:
                import os
                env_value = os.getenv(provider_config.auth_env)
                if env_value:
                    # Show only first few chars for security
                    masked_value = env_value[:8] + "..." if len(env_value) > 8 else env_value
                    self.console.print(f"    {provider_config.auth_env}: [green]✅ {masked_value}[/green]")
                else:
                    self.console.print(f"    {provider_config.auth_env}: [red]❌ Not found[/red]")
            elif provider_config.api_key:
                masked_key = provider_config.api_key[:8] + "..." if len(provider_config.api_key) > 8 else provider_config.api_key
                self.console.print(f"    Direct API Key: [green]✅ {masked_key}[/green]")
            else:
                self.console.print(f"    Authentication: [yellow]None required[/yellow]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()

    def switch_to_preset(self, llm_client: 'LLMClient', config, preset_name: str):
        """Switch to a specific configuration."""
        # Update configuration
        config.current_preset = preset_name
        
        # Apply preset to workflows
        preset_config = config.presets[preset_name]
        config.workflows = {
            "processing": preset_config["processing"],
            "conversation": preset_config["conversation"], 
            "jira_matching": preset_config["jira_matching"]
        }
        
        # Save configuration
        llm_client.storage.save_config(config)
        
        self.console.print(f"[green]✅ Switched to configuration: {preset_name}[/green]")
        
        # Test connection with new preset
        self.console.print("[dim]Testing new configuration...[/dim]")
        if self.test_model_connection(llm_client, silent=True):
            self.console.print("[green]✅ Connection test successful![/green]")
        else:
            self.console.print("[yellow]⚠️  Connection test failed - check your setup[/yellow]")
    
    def manage_entities(self):
        """Manage entities (collaborators, projects, tags)."""
        entity_registry = self.storage.entity_registry
        
        while True:
            # Clear screen for each menu display
            self.console.clear()
            
            self.console.print(f"\n[bold cyan]📊 Entity Management[/bold cyan]")
            
            # Show help if toggled on
            if self.help_visible:
                help_text = """
[bold]Entity Management Help:[/bold]

[green]What are entities?[/green]
Entities are the people, projects, and tags extracted from your journal entries.
They help organize and categorize your work for better tracking and reporting.

[green]Available Operations:[/green]
• [cyan]View by Type (1-3)[/cyan]: Browse and edit collaborators, projects, or tags
  - Edit names and add aliases for consistency
  - Merge duplicate entries automatically
  - See usage counts and recent mentions

• [cyan]AI Discovery (4)[/cyan]: Find new entities from your journal entries
  - Scans all entries for potential new collaborators, projects, tags
  - AI suggests entities based on context and patterns
  - Helps keep your entity registry up-to-date

• [cyan]Manual Addition (5)[/cyan]: Add entities directly
  - Useful for known entities not yet mentioned in entries
  - Add aliases during creation for better matching

• [cyan]Cleanup (6)[/cyan]: Remove unused or outdated entities
  - Find entities with zero usage counts
  - Safely remove entities no longer relevant

[green]Pro Tips:[/green]
• Entities are auto-extracted when processing journal entries
• Consistent naming improves reporting and 1:1 summaries
• Use aliases to handle name variations (e.g., "John" → "John Smith")
• Regular cleanup keeps your entity registry organized
                """
                self.console.print(Panel(help_text.strip(), title="Entity Management Help", border_style="blue"))
                self.console.print()
            
            # Show entity counts
            entity_counts = {
                'collaborator': len([e for e in entity_registry.entities.values() if e.type == 'collaborator']),
                'project': len([e for e in entity_registry.entities.values() if e.type == 'project']), 
                'tag': len([e for e in entity_registry.entities.values() if e.type == 'tag'])
            }
            
            stats_text = f"""
[bold]Entity Statistics:[/bold]
  👥 Collaborators: {entity_counts['collaborator']}
  📁 Projects: {entity_counts['project']}
  🏷️  Tags: {entity_counts['tag']}
            """
            
            self.console.print(Panel(stats_text.strip(), title="Current Entities", border_style="cyan"))
            
            # Show options
            help_text = "Help (hide)" if self.help_visible else "Help (show)"
            self.console.print("\n[bold]Options:[/bold]")
            self.console.print("  [cyan]1.[/cyan] View collaborators")
            self.console.print("  [cyan]2.[/cyan] View projects")
            self.console.print("  [cyan]3.[/cyan] View tags")
            self.console.print("  [cyan]4.[/cyan] AI-powered entity discovery from entries")
            self.console.print("  [cyan]5.[/cyan] Add new entity manually")
            self.console.print("  [cyan]6.[/cyan] Cleanup unused entities")
            self.console.print(f"  [cyan]h.[/cyan] {help_text}")
            self.console.print("  [cyan]q.[/cyan] Back to main menu")
            
            # Get user input
            self.console.print("\n[cyan bold]ENTITIES>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            choice = choice.lower()
            
            if choice == '1':
                self._show_entities_by_type('collaborator', '👥')
            elif choice == '2':
                self._show_entities_by_type('project', '📁')
            elif choice == '3':
                self._show_entities_by_type('tag', '🏷️')
            elif choice == '4':
                self._ai_entity_discovery()
            elif choice == '5':
                self._add_entity_manually()
            elif choice == '6':
                self._cleanup_entities()
            elif choice == 'h':
                self.help_visible = not self.help_visible
                # Menu will redraw with help toggled on next iteration
            else:
                self.console.print("[red]Invalid choice. Use 1-6, h for help, or 'q'.[/red]")
    
    def _show_entities_by_type(self, entity_type: str, icon: str):
        """Show entities of a specific type with editing options."""
        entity_registry = self.storage.entity_registry
        entities = [e for e in entity_registry.entities.values() if e.type == entity_type]
        
        if not entities:
            self.console.print(f"[yellow]No {entity_type}s found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        while True:
            # Sort by usage count (most used first)
            entities.sort(key=lambda e: e.usage_count, reverse=True)
            
            self.console.print(f"\n[bold cyan]{icon} {entity_type.title()}s ({len(entities)} total)[/bold cyan]")
            
            # Show help if toggled on
            if self.help_visible:
                help_text = f"""
[bold]{entity_type.title()} Management Help:[/bold]

[green]What you see:[/green]
• Entities sorted by usage count (most used first)
• Usage statistics show how often each entity appears in your entries
• Last used date helps identify stale entities
• Aliases show alternative names that map to the same entity

[green]Available Actions:[/green]
• [cyan]Select Number (1-{len(entities)})[/cyan]: Edit the entity
  - Change the canonical name
  - Add or remove aliases  
  - Delete the entity (with confirmation)

• [yellow]Add New (a)[/yellow]: Create a new {entity_type} manually
  - Useful for entities not yet mentioned in entries
  - Can add aliases during creation

• [yellow]Merge Entities (m)[/yellow]: Combine duplicate entities
  - Select primary and secondary entities
  - Usage counts are combined
  - All aliases are merged automatically

[green]Pro Tips:[/green]
• High usage count entities are likely important for reports
• Entities with zero usage might be candidates for cleanup
• Use aliases to handle name variations consistently
• Merging entities helps consolidate duplicate entries
                """
                self.console.print(Panel(help_text.strip(), title=f"{entity_type.title()} Help", border_style="blue"))
                self.console.print()
            
            for i, entity in enumerate(entities, 1):
                last_used = entity.last_used if entity.last_used else "Never"
                aliases_text = f" (aliases: {', '.join(entity.aliases)})" if entity.aliases else ""
                self.console.print(f"  [cyan]{i:2d}.[/cyan] {entity.canonical_name}{aliases_text}")
                self.console.print(f"      [dim]Used {entity.usage_count} times, last: {last_used}[/dim]")
            
            help_text = "Help (hide)" if self.help_visible else "Help (show)"
            self.console.print(f"\n[dim]Options: 1-{len(entities)} to edit, [yellow]a[/yellow]=add new, [yellow]m[/yellow]=merge entities, [cyan]h[/cyan]={help_text.lower()}, [yellow]q[/yellow]=back[/dim]")
            self.console.print(f"\n[cyan bold]{entity_type.upper()}>[/cyan bold] ", end="")
            choice = self.safe_input()
            
            if choice is None or choice.lower() == 'q':
                break
            elif choice.lower() == 'a':
                self._add_entity_manually(entity_type)
                # Refresh entities list
                entities = [e for e in entity_registry.entities.values() if e.type == entity_type]
            elif choice.lower() == 'm':
                self._merge_entities_interactive(entity_type, entities)
                # Refresh entities list
                entities = [e for e in entity_registry.entities.values() if e.type == entity_type]
            elif choice.lower() == 'h':
                self.help_visible = not self.help_visible
                # Interface will redraw with help toggled on next iteration
            else:
                try:
                    entity_num = int(choice)
                    if 1 <= entity_num <= len(entities):
                        entity = entities[entity_num - 1]
                        if self._edit_entity_interactive(entity):
                            # Refresh entities list
                            entities = [e for e in entity_registry.entities.values() if e.type == entity_type]
                    else:
                        self.console.print(f"[red]Invalid selection. Choose 1-{len(entities)}[/red]")
                except ValueError:
                    self.console.print("[red]Invalid input. Enter a number, 'a', 'm', 'h', or 'q'.[/red]")
    
    def _cleanup_entities(self):
        """Clean up unused entities."""
        entity_registry = self.storage.entity_registry
        
        # Find entities with very low usage
        unused_entities = [e for e in entity_registry.entities.values() if e.usage_count <= 1]
        
        if not unused_entities:
            self.console.print("[green]No unused entities found. Registry is clean![/green]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        self.console.print(f"\n[yellow]Found {len(unused_entities)} entities with minimal usage (≤1 use):[/yellow]")
        
        for entity in unused_entities:
            icon = {'collaborator': '👥', 'project': '📁', 'tag': '🏷️'}.get(entity.type, '📝')
            self.console.print(f"  {icon} {entity.canonical_name} ({entity.type}, used {entity.usage_count} times)")
        
        self.console.print(f"\n[bold yellow]Remove these {len(unused_entities)} entities? (y/n)>[/bold yellow] ", end="")
        confirm = self.safe_input()
        
        if confirm and confirm.lower() == 'y':
            entity_registry.cleanup_entities(min_usage=2)
            entity_registry.save_entities()
            self.console.print(f"[green]✅ Removed {len(unused_entities)} unused entities.[/green]")
        else:
            self.console.print("[dim]Cleanup cancelled.[/dim]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _edit_entity_interactive(self, entity) -> bool:
        """Interactive entity editing. Returns True if entity was modified."""
        entity_registry = self.storage.entity_registry
        
        while True:
            self.console.print(f"\n[bold cyan]✏️  Editing {entity.type}: {entity.canonical_name}[/bold cyan]")
            self.console.print(f"[dim]ID: {entity.id}[/dim]")
            self.console.print(f"[dim]Used {entity.usage_count} times, last: {entity.last_used or 'Never'}[/dim]")
            
            if entity.aliases:
                self.console.print(f"[dim]Aliases: {', '.join(entity.aliases)}[/dim]")
            
            self.console.print("\n[bold]Options:[/bold]")
            self.console.print("  [cyan]1.[/cyan] Edit name")
            self.console.print("  [cyan]2.[/cyan] Edit aliases")
            self.console.print("  [cyan]3.[/cyan] Delete this entity")
            self.console.print("  [cyan]q.[/cyan] Back")
            
            self.console.print(f"\n[cyan bold]EDIT>[/cyan bold] ", end="")
            choice = self.safe_input()
            
            if choice is None or choice.lower() == 'q':
                return False
            elif choice == '1':
                self.console.print(f"\n[bold]Current name:[/bold] {entity.canonical_name}")
                self.console.print(f"[cyan bold]New name (or Enter to keep current)>[/cyan bold] ", end="")
                new_name = self.safe_input()
                if new_name and new_name.strip():
                    entity_registry.update_entity(entity.id, canonical_name=new_name.strip())
                    entity_registry.save_entities()
                    self.console.print("[green]✅ Name updated[/green]")
                    return True
            elif choice == '2':
                current_aliases = list(entity.aliases) if entity.aliases else []
                self.console.print(f"\n[bold]Current aliases:[/bold] {', '.join(current_aliases) if current_aliases else 'None'}")
                self.console.print("[dim]Enter aliases separated by commas (or Enter to keep current):[/dim]")
                self.console.print(f"[cyan bold]Aliases>[/cyan bold] ", end="")
                aliases_input = self.safe_input()
                if aliases_input is not None:
                    if aliases_input.strip():
                        new_aliases = [alias.strip() for alias in aliases_input.split(',') if alias.strip()]
                    else:
                        new_aliases = []
                    entity_registry.update_entity(entity.id, aliases=new_aliases)
                    entity_registry.save_entities()
                    self.console.print("[green]✅ Aliases updated[/green]")
                    return True
            elif choice == '3':
                self.console.print(f"\n[red bold]Delete '{entity.canonical_name}'? This cannot be undone! (y/N)>[/red bold] ", end="")
                confirm = self.safe_input()
                if confirm and confirm.lower() == 'y':
                    entity_registry.delete_entity(entity.id)
                    entity_registry.save_entities()
                    self.console.print("[green]✅ Entity deleted[/green]")
                    return True
                else:
                    self.console.print("[dim]Delete cancelled[/dim]")
            else:
                self.console.print("[red]Invalid choice. Use 1-3 or 'q'.[/red]")
    
    def _add_entity_manually(self, entity_type: str = None):
        """Add a new entity manually."""
        entity_registry = self.storage.entity_registry
        
        if entity_type is None:
            self.console.print("\n[bold cyan]➕ Add New Entity[/bold cyan]")
            self.console.print("\n[bold]Entity type:[/bold]")
            self.console.print("  [cyan]1.[/cyan] Collaborator")
            self.console.print("  [cyan]2.[/cyan] Project") 
            self.console.print("  [cyan]3.[/cyan] Tag")
            
            self.console.print(f"\n[cyan bold]TYPE>[/cyan bold] ", end="")
            type_choice = self.safe_input()
            
            if type_choice == '1':
                entity_type = 'collaborator'
            elif type_choice == '2':
                entity_type = 'project'
            elif type_choice == '3':
                entity_type = 'tag'
            else:
                self.console.print("[red]Invalid choice. Cancelled.[/red]")
                return
        
        icon = {'collaborator': '👥', 'project': '📁', 'tag': '🏷️'}.get(entity_type, '📝')
        
        self.console.print(f"\n[bold cyan]{icon} Add New {entity_type.title()}[/bold cyan]")
        self.console.print(f"[cyan bold]Name>[/cyan bold] ", end="")
        name = self.safe_input()
        
        if not name or not name.strip():
            self.console.print("[red]Name cannot be empty. Cancelled.[/red]")
            return
        
        name = name.strip()
        
        # Check for duplicates
        existing_matches = entity_registry.find_entity_matches(name, entity_type)
        if existing_matches and existing_matches[0].confidence > 0.9:
            self.console.print(f"[yellow]⚠️  Similar entity already exists: '{existing_matches[0].canonical_name}'[/yellow]")
            self.console.print(f"[cyan bold]Continue anyway? (y/N)>[/cyan bold] ", end="")
            confirm = self.safe_input()
            if not confirm or confirm.lower() != 'y':
                self.console.print("[dim]Cancelled[/dim]")
                return
        
        # Get aliases
        self.console.print("[dim]Aliases (comma-separated, optional):[/dim]")
        self.console.print(f"[cyan bold]Aliases>[/cyan bold] ", end="")
        aliases_input = self.safe_input()
        
        aliases = []
        if aliases_input and aliases_input.strip():
            aliases = [alias.strip() for alias in aliases_input.split(',') if alias.strip()]
        
        # Add entity
        entity_id = entity_registry.add_or_update_entity(name, entity_type, aliases)
        entity_registry.save_entities()
        
        self.console.print(f"[green]✅ Added {entity_type} '{name}' (ID: {entity_id})[/green]")
    
    def _merge_entities_interactive(self, entity_type: str, entities):
        """Interactive entity merging."""
        entity_registry = self.storage.entity_registry
        
        if len(entities) < 2:
            self.console.print("[yellow]Need at least 2 entities to merge.[/yellow]")
            return
        
        self.console.print(f"\n[bold cyan]🔗 Merge {entity_type.title()}s[/bold cyan]")
        self.console.print("Select the PRIMARY entity (the one to keep):")
        
        for i, entity in enumerate(entities, 1):
            aliases_text = f" (aliases: {', '.join(entity.aliases)})" if entity.aliases else ""
            self.console.print(f"  [cyan]{i:2d}.[/cyan] {entity.canonical_name}{aliases_text}")
            self.console.print(f"      [dim]Used {entity.usage_count} times[/dim]")
        
        self.console.print(f"\n[cyan bold]PRIMARY (1-{len(entities)})>[/cyan bold] ", end="")
        primary_choice = self.safe_input()
        
        try:
            primary_num = int(primary_choice)
            if not (1 <= primary_num <= len(entities)):
                raise ValueError()
        except (ValueError, TypeError):
            self.console.print("[red]Invalid selection. Cancelled.[/red]")
            return
        
        primary_entity = entities[primary_num - 1]
        
        self.console.print(f"\nPrimary: [bold]{primary_entity.canonical_name}[/bold]")
        self.console.print("Select the SECONDARY entity (the one to merge into primary):")
        
        for i, entity in enumerate(entities, 1):
            if i == primary_num:
                continue
            aliases_text = f" (aliases: {', '.join(entity.aliases)})" if entity.aliases else ""
            self.console.print(f"  [cyan]{i:2d}.[/cyan] {entity.canonical_name}{aliases_text}")
            self.console.print(f"      [dim]Used {entity.usage_count} times[/dim]")
        
        self.console.print(f"\n[cyan bold]SECONDARY (1-{len(entities)}, not {primary_num})>[/cyan bold] ", end="")
        secondary_choice = self.safe_input()
        
        try:
            secondary_num = int(secondary_choice)
            if not (1 <= secondary_num <= len(entities)) or secondary_num == primary_num:
                raise ValueError()
        except (ValueError, TypeError):
            self.console.print("[red]Invalid selection. Cancelled.[/red]")
            return
        
        secondary_entity = entities[secondary_num - 1]
        
        # Confirm merge
        self.console.print(f"\n[bold yellow]Merge '{secondary_entity.canonical_name}' into '{primary_entity.canonical_name}'?[/bold yellow]")
        self.console.print(f"[dim]- '{secondary_entity.canonical_name}' will be deleted[/dim]")
        self.console.print(f"[dim]- Usage counts will be combined ({primary_entity.usage_count} + {secondary_entity.usage_count})[/dim]")
        self.console.print(f"[dim]- All aliases will be merged[/dim]")
        self.console.print(f"[red bold]Confirm merge? (y/N)>[/red bold] ", end="")
        
        confirm = self.safe_input()
        if confirm and confirm.lower() == 'y':
            if entity_registry.merge_entities(primary_entity.id, secondary_entity.id):
                entity_registry.save_entities()
                self.console.print("[green]✅ Entities merged successfully[/green]")
            else:
                self.console.print("[red]❌ Merge failed[/red]")
        else:
            self.console.print("[dim]Merge cancelled[/dim]")
    
    def _ai_entity_discovery(self):
        """Use AI to discover entities from existing journal entries for approval and editing."""
        self.console.print("\n[bold cyan]🤖 AI-Powered Entity Discovery[/bold cyan]")
        self.console.print("This will analyze your existing journal entries to find potential entities.")
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("  [cyan]1.[/cyan] Discover collaborators from all entries")
        self.console.print("  [cyan]2.[/cyan] Discover projects from all entries")
        self.console.print("  [cyan]3.[/cyan] Discover all entity types (comprehensive)")
        self.console.print("  [cyan]q.[/cyan] Back")
        
        self.console.print(f"\n[cyan bold]DISCOVER>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if choice is None or choice.lower() == 'q':
            return
        elif choice == '1':
            self._ai_discover_by_type('collaborator')
        elif choice == '2':
            self._ai_discover_by_type('project')
        elif choice == '3':
            self._ai_discover_comprehensive()
        else:
            self.console.print("[red]Invalid choice. Use 1-3 or 'q'.[/red]")
    
    def _ai_discover_by_type(self, entity_type: str):
        """Discover entities of a specific type using AI."""
        # Load recent entries to analyze
        entries = self.storage.load_all_entries(limit=50)  # Analyze last 50 entries
        
        if not entries:
            self.console.print("[yellow]No entries found to analyze.[/yellow]")
            return
        
        # Prepare text for AI analysis
        entry_texts = []
        for entry in entries:
            entry_texts.append(f"Date: {entry.date}\nText: {entry.raw_input}")
        
        combined_text = "\n\n---\n\n".join(entry_texts)
        
        icon = {'collaborator': '👥', 'project': '📁', 'tag': '🏷️'}.get(entity_type, '📝')
        model_info = self._get_current_model_info("processing")
        
        self.console.print(f"\n[cyan]🤖⏳ Analyzing {len(entries)} entries with {model_info} to find {entity_type}s...[/cyan]")
        
        # Create AI prompt for entity discovery
        system_prompt = f"""You are analyzing work journal entries to discover {entity_type}s that should be tracked.

For {entity_type}s, extract:
- Names that appear multiple times across entries
- Clear, distinct entities that represent actual {entity_type}s
- Standardized names (not variations like "John" and "John Smith" - pick one canonical form)

CRITICAL INSTRUCTIONS:
- Only extract {entity_type}s that are explicitly mentioned in the text
- Focus on entities that appear multiple times or seem important
- For collaborators: Extract actual person names, not roles or departments
- For projects: Extract specific project names with concrete deliverables
- For tags: Extract technology, process, or domain-specific terms
- Provide a confidence score (1-5) for each entity
- Include a brief explanation of why each entity should be tracked

Return ONLY valid JSON in this format:
{{
    "discovered_entities": [
        {{
            "name": "Entity Name",
            "confidence": 4,
            "occurrences": 3,
            "explanation": "Why this entity should be tracked",
            "example_contexts": ["Brief context 1", "Brief context 2"]
        }}
    ]
}}"""

        user_prompt = f"Analyze these journal entries and discover {entity_type}s:\n\n{combined_text}"
        
        try:
            from .entry_processor import EntryProcessor
            processor = EntryProcessor(storage=self.storage)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            with self.llm_spinner(f"🤖 Discovering {entity_type}s", "processing"):
                response = processor.llm_client.call_llm("processing", messages, max_tokens=2000, temperature=0.1)
            
            # Parse the AI response
            import json
            discovery_data = json.loads(response)
            discovered_entities = discovery_data.get("discovered_entities", [])
            
            if not discovered_entities:
                self.console.print(f"[yellow]No {entity_type}s discovered in your entries.[/yellow]")
                return
            
            # Show discovered entities for approval
            self._approve_discovered_entities(discovered_entities, entity_type, icon)
            
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Failed to parse AI response: {e}[/red]")
        except Exception as e:
            self.console.print(f"[red]Entity discovery failed: {e}[/red]")
    
    def _ai_discover_comprehensive(self):
        """Comprehensive AI entity discovery for all types."""
        self.console.print(f"\n[yellow]🤖⏳ Running comprehensive entity discovery...[/yellow]")
        
        for entity_type in ['collaborator', 'project', 'tag']:
            icon = {'collaborator': '👥', 'project': '📁', 'tag': '🏷️'}[entity_type]
            self.console.print(f"\n[dim]Discovering {icon} {entity_type}s...[/dim]")
            self._ai_discover_by_type(entity_type)
        
        self.console.print(f"\n[green]✅ Comprehensive entity discovery complete![/green]")
    
    def _approve_discovered_entities(self, discovered_entities, entity_type: str, icon: str):
        """Show discovered entities for user approval and editing."""
        entity_registry = self.storage.entity_registry
        
        self.console.print(f"\n[bold cyan]{icon} Discovered {entity_type.title()}s[/bold cyan]")
        self.console.print(f"Found {len(discovered_entities)} potential {entity_type}s for your review:\n")
        
        approved_entities = []
        
        for i, entity_data in enumerate(discovered_entities, 1):
            name = entity_data.get('name', 'Unknown')
            confidence = entity_data.get('confidence', 0)
            occurrences = entity_data.get('occurrences', 0)
            explanation = entity_data.get('explanation', 'No explanation provided')
            examples = entity_data.get('example_contexts', [])
            
            # Check if entity already exists
            existing_matches = entity_registry.find_entity_matches(name, entity_type)
            is_duplicate = existing_matches and existing_matches[0].confidence > 0.8
            
            # Display entity info
            confidence_color = "green" if confidence >= 4 else "yellow" if confidence >= 3 else "red"
            duplicate_text = " [dim red](possible duplicate)[/dim red]" if is_duplicate else ""
            
            self.console.print(f"[bold]{i}. {name}{duplicate_text}[/bold]")
            self.console.print(f"   [dim]Confidence: [{confidence_color}]{confidence}/5[/{confidence_color}], Occurrences: {occurrences}[/dim]")
            self.console.print(f"   [dim]{explanation}[/dim]")
            
            if examples:
                self.console.print(f"   [dim]Examples: {', '.join(examples[:2])}[/dim]")
            
            if is_duplicate:
                self.console.print(f"   [yellow]⚠️  Similar to existing: {existing_matches[0].canonical_name}[/yellow]")
            
            # Ask for approval
            self.console.print(f"\n   [cyan bold]Add '{name}' to {entity_type}s? (y/n/e=edit name)>[/cyan bold] ", end="")
            decision = self.safe_input()
            
            if decision and decision.lower() == 'y':
                approved_entities.append({'name': name, 'aliases': []})
                self.console.print(f"   [green]✅ Approved[/green]")
            elif decision and decision.lower() == 'e':
                self.console.print(f"   [cyan bold]Enter corrected name>[/cyan bold] ", end="")
                corrected_name = self.safe_input()
                if corrected_name and corrected_name.strip():
                    self.console.print(f"   [cyan bold]Aliases (comma-separated, optional)>[/cyan bold] ", end="")
                    aliases_input = self.safe_input()
                    aliases = [a.strip() for a in aliases_input.split(',') if a.strip()] if aliases_input else []
                    aliases.append(name)  # Add original name as alias
                    approved_entities.append({'name': corrected_name.strip(), 'aliases': aliases})
                    self.console.print(f"   [green]✅ Approved as '{corrected_name}'[/green]")
            else:
                self.console.print(f"   [dim]⏭️  Skipped[/dim]")
            
            self.console.print()  # Add spacing
        
        # Batch add approved entities
        if approved_entities:
            self.console.print(f"\n[cyan]➕ Adding {len(approved_entities)} approved {entity_type}s...[/cyan]")
            for entity_data in approved_entities:
                entity_id = entity_registry.add_or_update_entity(
                    entity_data['name'], 
                    entity_type, 
                    entity_data['aliases']
                )
                self.console.print(f"   [green]✅[/green] {entity_data['name']} (ID: {entity_id})")
            
            entity_registry.save_entities()
            self.console.print(f"\n[green]🎉 Successfully added {len(approved_entities)} new {entity_type}s![/green]")
        else:
            self.console.print(f"[yellow]No {entity_type}s were approved for addition.[/yellow]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def data_management(self):
        """Comprehensive data management interface."""
        while True:
            # Clear screen for each menu display
            self.console.clear()
            
            # Get current data statistics
            stats = self.storage.get_data_statistics()
            recycle_stats = self.storage.get_recycle_bin_stats()
            
            self.console.print(f"\n[bold cyan]💾 Data Management[/bold cyan]")
            
            # Display data overview
            overview_text = f"""
[bold]Data Overview:[/bold]
  📝 Entries: {stats['entries']['total_entries']} entries across {stats['entries']['total_files']} files
  📅 Date Range: {stats['entries']['date_range']['earliest'] or 'None'} to {stats['entries']['date_range']['latest'] or 'None'}
  👥 Entities: {stats['entities']['collaborators']} collaborators, {stats['entities']['projects']} projects, {stats['entities']['tags']} tags
  🗑️  Recycle Bin: {recycle_stats['total_count']} deleted entries
  💾 Backups: {stats['backups']['total_backups']} backups ({stats['backups']['latest_backup'] or 'None'})
  💽 Storage: {stats['storage']['total_size_mb']} MB at {stats['storage']['base_path']}
            """
            
            self.console.print(Panel(overview_text.strip(), title="System Status", border_style="cyan"))
            
            # Show options
            self.console.print("\n[bold]Data Management Options:[/bold]")
            self.console.print("  [cyan]1.[/cyan] Create backup")
            self.console.print("  [cyan]2.[/cyan] View backups")
            self.console.print("  [cyan]3.[/cyan] Restore from backup")
            self.console.print("  [cyan]4.[/cyan] Reprocess entries (apply current LLM logic)")
            self.console.print("  [cyan]5.[/cyan] System statistics")
            recycle_color = "yellow" if recycle_stats['total_count'] > 0 else "dim cyan"
            self.console.print(f"  [{recycle_color}]6.[/{recycle_color}] Recycle bin (empty trash)")
            self.console.print("  [cyan]q.[/cyan] Back to main menu")
            
            # Get user input
            self.console.print("\n[cyan bold]DATA>[/cyan bold] ", end="")
            choice = self.safe_input()
            if choice is None or choice.lower() == 'q':
                break
            choice = choice.lower()
            
            if choice == '1':
                self._create_backup_interactive()
            elif choice == '2':
                self._view_backups_interactive()
            elif choice == '3':
                self._restore_backup_interactive()
            elif choice == '4':
                self._reprocess_entries_interactive()
            elif choice == '5':
                self._show_detailed_statistics()
            elif choice == '6':
                self._manage_recycle_bin_interactive()
            else:
                self.console.print("[red]Invalid choice. Use 1-6 or 'q'.[/red]")
    
    def _create_backup_interactive(self):
        """Interactive backup creation."""
        self.console.print("\n[bold cyan]💾 Create Backup[/bold cyan]")
        
        # Suggest backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"manual_{timestamp}"
        
        self.console.print(f"[cyan bold]Backup name (or Enter for '{suggested_name}')>[/cyan bold] ", end="")
        backup_name = self.safe_input()
        
        if not backup_name or not backup_name.strip():
            backup_name = suggested_name
        else:
            backup_name = backup_name.strip()
        
        self.console.print(f"\n[cyan]🔄 Creating backup '{backup_name}'...[/cyan]")
        
        try:
            backup_path = self.storage.create_backup(backup_name)
            backup_size = Path(backup_path).stat().st_size / (1024 * 1024)  # MB
            
            self.console.print(f"[green]✅ Backup created successfully![/green]")
            self.console.print(f"[dim]Location: {backup_path}[/dim]")
            self.console.print(f"[dim]Size: {backup_size:.2f} MB[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Backup failed: {e}[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _view_backups_interactive(self):
        """Interactive backup viewing."""
        self.console.print("\n[bold cyan]📋 Available Backups[/bold cyan]")
        
        backups = self.storage.list_backups()
        
        if not backups:
            self.console.print("[yellow]No backups found.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        # Display backups
        for i, backup in enumerate(backups, 1):
            backup_type = backup.get("backup_type", "unknown")
            
            if backup_type == "corrupted":
                self.console.print(f"[red]{i:2d}. {backup['backup_name']} [CORRUPTED][/red]")
                self.console.print(f"    [dim red]Error: {backup.get('error', 'Unknown error')}[/dim red]")
            else:
                # Parse creation date for display
                try:
                    created_at = datetime.fromisoformat(backup["created_at"].replace('Z', '+00:00'))
                    created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    created_str = backup["created_at"]
                
                file_size = backup.get("file_size", 0) / (1024 * 1024)  # MB
                entries = backup.get("total_entries", "unknown")
                entities = backup.get("total_entities", "unknown")
                
                icon = "🟢" if backup_type == "full" else "🟡"
                self.console.print(f"{icon} [cyan]{i:2d}.[/cyan] [bold]{backup['backup_name']}[/bold]")
                self.console.print(f"    [dim]Created: {created_str} | Size: {file_size:.2f} MB[/dim]")
                self.console.print(f"    [dim]Contains: {entries} entries, {entities} entities[/dim]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _restore_backup_interactive(self):
        """Interactive backup restoration."""
        self.console.print("\n[bold red]⚠️  RESTORE FROM BACKUP[/bold red]")
        self.console.print("[yellow]WARNING: This will REPLACE all current data![/yellow]")
        
        backups = self.storage.list_backups()
        
        if not backups:
            self.console.print("[yellow]No backups available for restore.[/yellow]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        # Show available backups
        valid_backups = [b for b in backups if b.get("backup_type") != "corrupted"]
        
        if not valid_backups:
            self.console.print("[red]No valid backups available for restore.[/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        self.console.print(f"\n[bold]Available Backups:[/bold]")
        for i, backup in enumerate(valid_backups, 1):
            try:
                created_at = datetime.fromisoformat(backup["created_at"].replace('Z', '+00:00'))
                created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
            except:
                created_str = backup["created_at"]
            
            self.console.print(f"  [cyan]{i}.[/cyan] {backup['backup_name']} ({created_str})")
        
        self.console.print(f"\n[cyan bold]Select backup to restore (1-{len(valid_backups)}) or 'q' to cancel>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if not choice or choice.lower() == 'q':
            self.console.print("[dim]Restore cancelled.[/dim]")
            return
        
        try:
            backup_num = int(choice)
            if not (1 <= backup_num <= len(valid_backups)):
                raise ValueError()
        except ValueError:
            self.console.print("[red]Invalid selection. Restore cancelled.[/red]")
            return
        
        selected_backup = valid_backups[backup_num - 1]
        
        # Final confirmation
        self.console.print(f"\n[red bold]FINAL WARNING: Restore '{selected_backup['backup_name']}'?[/red bold]")
        self.console.print("[red]This will PERMANENTLY REPLACE all current entries, entities, and configuration![/red]")
        self.console.print("[yellow]A safety backup will be created automatically.[/yellow]")
        self.console.print(f"\n[red bold]Type 'RESTORE' to confirm>[/red bold] ", end="")
        
        confirm = self.safe_input()
        if confirm != "RESTORE":
            self.console.print("[dim]Restore cancelled - confirmation not received.[/dim]")
            return
        
        # Perform restore
        self.console.print(f"\n[yellow]🔄 Restoring from '{selected_backup['backup_name']}'...[/yellow]")
        
        try:
            success = self.storage.restore_from_backup(selected_backup["backup_name"], confirm=True)
            if success:
                self.console.print("[green]✅ Restore completed successfully![/green]")
                self.console.print("[yellow]⚠️  You may need to restart the application for all changes to take effect.[/yellow]")
            else:
                self.console.print("[red]❌ Restore failed for unknown reason.[/red]")
        except Exception as e:
            self.console.print(f"[red]❌ Restore failed: {e}[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _reprocess_entries_interactive(self):
        """Interactive entry reprocessing."""
        self.console.print("\n[bold cyan]🔄 Reprocess Entries[/bold cyan]")
        self.console.print("This will re-run current LLM processing logic on existing entries.")
        self.console.print("[yellow]⚠️  This will modify your existing data![/yellow]")
        
        # Show processing options
        self.console.print("\n[bold]Processing Options:[/bold]")
        self.console.print("  [cyan]1.[/cyan] Reprocess all entries (up to 1000)")
        self.console.print("  [cyan]2.[/cyan] Reprocess recent entries (last 30 days)")
        self.console.print("  [cyan]3.[/cyan] Reprocess specific date (YYYY-MM-DD)")
        self.console.print("  [cyan]q.[/cyan] Cancel")
        
        self.console.print(f"\n[cyan bold]Select processing scope>[/cyan bold] ", end="")
        choice = self.safe_input()
        
        if not choice or choice.lower() == 'q':
            self.console.print("[dim]Reprocessing cancelled.[/dim]")
            return
        
        # Determine filter
        entry_filter = "all"
        filter_description = "all entries"
        
        if choice == '2':
            entry_filter = "recent"
            filter_description = "recent entries (last 30 days)"
        elif choice == '3':
            self.console.print(f"\n[cyan bold]Enter date (YYYY-MM-DD)>[/cyan bold] ", end="")
            date_input = self.safe_input()
            if not date_input:
                self.console.print("[red]No date provided. Cancelled.[/red]")
                return
            entry_filter = date_input.strip()
            filter_description = f"entries from {entry_filter}"
        elif choice != '1':
            self.console.print("[red]Invalid choice. Cancelled.[/red]")
            return
        
        # Create backup before reprocessing
        self.console.print(f"\n[yellow]🔄 Creating backup before reprocessing...[/yellow]")
        try:
            backup_name = f"pre_reprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.storage.create_backup(backup_name)
            self.console.print(f"[green]✅ Backup created: {backup_name}[/green]")
        except Exception as e:
            self.console.print(f"[red]❌ Backup failed: {e}[/red]")
            self.console.print("[red]Reprocessing cancelled for safety.[/red]")
            return
        
        # Final confirmation
        self.console.print(f"\n[yellow]Ready to reprocess {filter_description}[/yellow]")
        self.console.print(f"[cyan bold]Continue? (y/N)>[/cyan bold] ", end="")
        confirm = self.safe_input()
        
        if not confirm or confirm.lower() != 'y':
            self.console.print("[dim]Reprocessing cancelled.[/dim]")
            return
        
        # Start reprocessing
        model_info = self._get_current_model_info("processing")
        self.console.print(f"\n[cyan]🤖⏳ Reprocessing with {model_info}...[/cyan]")
        
        try:
            result = self.storage.reprocess_entries(entry_filter=entry_filter, batch_size=5)
            
            if result["status"] == "no_entries":
                self.console.print("[yellow]No entries found to reprocess.[/yellow]")
            elif result["status"] == "completed":
                self.console.print(f"[green]✅ Reprocessing completed![/green]")
                self.console.print(f"[dim]Total entries: {result['total_entries']}[/dim]")
                self.console.print(f"[dim]Successfully processed: {result['processed']}[/dim]")
                
                if result["errors"] > 0:
                    self.console.print(f"[yellow]⚠️  Errors encountered: {result['errors']}[/yellow]")
                    
                    # Show error details if any
                    if result.get("error_details"):
                        self.console.print("\n[red]Error Details:[/red]")
                        for error in result["error_details"][:5]:  # Show first 5 errors
                            self.console.print(f"  [red]•[/red] {error['date']} - {error['error']}")
                        
                        if len(result["error_details"]) > 5:
                            self.console.print(f"  [dim]... and {len(result['error_details']) - 5} more errors[/dim]")
            else:
                self.console.print(f"[red]❌ Reprocessing failed with status: {result['status']}[/red]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Reprocessing failed: {e}[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _show_detailed_statistics(self):
        """Show detailed system statistics."""
        self.console.print("\n[bold cyan]📊 Detailed Statistics[/bold cyan]")
        
        stats = self.storage.get_data_statistics()
        
        # Entry statistics
        entries_panel = Panel(
            f"""[bold]Total Entries:[/bold] {stats['entries']['total_entries']}
[bold]Storage Files:[/bold] {stats['entries']['total_files']}
[bold]Date Range:[/bold] {stats['entries']['date_range']['earliest'] or 'None'} to {stats['entries']['date_range']['latest'] or 'None'}""",
            title="📝 Journal Entries",
            border_style="green"
        )
        self.console.print(entries_panel)
        
        # Entity statistics  
        entities_panel = Panel(
            f"""[bold]Collaborators:[/bold] {stats['entities']['collaborators']}
[bold]Projects:[/bold] {stats['entities']['projects']}
[bold]Tags:[/bold] {stats['entities']['tags']}
[bold]Total Entities:[/bold] {sum(stats['entities'].values())}""",
            title="👥 Entity Registry", 
            border_style="blue"
        )
        self.console.print(entities_panel)
        
        # Backup statistics
        backup_panel = Panel(
            f"""[bold]Total Backups:[/bold] {stats['backups']['total_backups']}
[bold]Latest Backup:[/bold] {stats['backups']['latest_backup'] or 'None'}""",
            title="💾 Backup System",
            border_style="yellow"
        )
        self.console.print(backup_panel)
        
        # Storage statistics
        storage_panel = Panel(
            f"""[bold]Base Directory:[/bold] {stats['storage']['base_path']}
[bold]Total Size:[/bold] {stats['storage']['total_size_mb']} MB""",
            title="💽 Storage Usage",
            border_style="magenta"
        )
        self.console.print(storage_panel)
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _manage_recycle_bin_interactive(self):
        """Interactive recycle bin management (empty trash)."""
        self.console.print("\n[bold red]🗑️  Recycle Bin Management[/bold red]")
        
        # Get recycle bin statistics
        recycle_stats = self.storage.get_recycle_bin_stats()
        
        if recycle_stats['total_count'] == 0:
            self.console.print("[green]✅ Recycle bin is empty - no deleted entries to manage.[/green]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        # Show recycle bin contents
        overview_text = f"""[bold]Deleted Entries Overview:[/bold]
  📝 Total Entries: {recycle_stats['total_count']}
  📅 Date Range: {recycle_stats['oldest_date'] or 'Unknown'} to {recycle_stats['newest_date'] or 'Unknown'}
  💽 Estimated Size: {recycle_stats['total_size_estimate']} bytes
  
[yellow]⚠️  These entries are soft-deleted and can be permanently removed.[/yellow]"""
        
        recycle_panel = Panel(overview_text.strip(), title="🗑️ Recycle Bin Status", border_style="red")
        self.console.print(recycle_panel)
        
        # Show options
        self.console.print("\n[bold]Recycle Bin Options:[/bold]")
        self.console.print("  [red]1.[/red] View deleted entries")
        self.console.print("  [red]2.[/red] Empty recycle bin (permanently delete all)")
        self.console.print("  [cyan]q.[/cyan] Back to data management")
        
        # Get user choice
        self.console.print("\n[red bold]RECYCLE>[/red bold] ", end="")
        choice = self.safe_input()
        if choice is None or choice.lower() == 'q':
            return
        choice = choice.lower()
        
        if choice == '1':
            self._view_deleted_entries()
        elif choice == '2':
            self._empty_recycle_bin_confirm()
        else:
            self.console.print("[red]Invalid choice. Use 1-2 or 'q'.[/red]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
    
    def _view_deleted_entries(self):
        """Show all deleted (tombstoned) entries."""
        self.console.print("\n[bold red]👁️  View Deleted Entries[/bold red]")
        
        tombstoned = self.storage.load_tombstoned_entries()
        
        if not tombstoned:
            self.console.print("[green]No deleted entries found.[/green]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        # Display each tombstoned entry
        for i, entry_data in enumerate(tombstoned, 1):
            date = entry_data.get('date', 'Unknown date')
            tombstoned_at = entry_data.get('tombstoned_at', 'Unknown time')
            raw_input = entry_data.get('raw_input', 'No content')
            entry_id = entry_data.get('id', 'Unknown ID')
            
            # Truncate content if too long
            if len(raw_input) > 200:
                raw_input = raw_input[:200] + "..."
            
            entry_info = f"""[bold]Entry {i}:[/bold]
  🆔 ID: {entry_id}
  📅 Original Date: {date}
  🗑️  Deleted: {tombstoned_at}
  📝 Content: {raw_input}
            """
            
            entry_panel = Panel(entry_info.strip(), title=f"Deleted Entry {i}", border_style="dim red")
            self.console.print(entry_panel)
        
        self.console.print(f"\n[dim]Showing {len(tombstoned)} deleted entries.[/dim]")
        self.console.print("[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def _empty_recycle_bin_confirm(self):
        """Confirm and permanently delete all tombstoned entries."""
        self.console.print("\n[bold red]⚠️  Empty Recycle Bin (Permanent Deletion)[/bold red]")
        
        recycle_stats = self.storage.get_recycle_bin_stats()
        
        self.console.print(f"[yellow]This will permanently delete {recycle_stats['total_count']} entries.[/yellow]")
        self.console.print("[red bold]⚠️  This action cannot be undone![/red bold]")
        self.console.print("\n[bold]Type 'EMPTY' to confirm permanent deletion, or anything else to cancel:[/bold]")
        
        self.console.print("[red bold]CONFIRM>[/red bold] ", end="")
        confirmation = self.safe_input()
        
        if confirmation != "EMPTY":
            self.console.print("[green]✅ Operation cancelled - no entries were deleted.[/green]")
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
            return
        
        # Perform the permanent deletion
        self.console.print("\n[red]🔄 Permanently deleting tombstoned entries...[/red]")
        
        try:
            result = self.storage.empty_recycle_bin()
            
            self.console.print(f"[green]✅ Recycle bin emptied successfully![/green]")
            self.console.print(f"[dim]Deleted {result['deleted_entries']} entries from {result['files_modified']} files.[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to empty recycle bin: {e}[/red]")
        
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        self.safe_input()
    
    def test_model_connection(self, llm_client: 'LLMClient', silent: bool = False) -> bool:
        """Test connection to current model configuration."""
        if not silent:
            self.console.print("\n[bold blue]🔍 Testing Model Configuration[/bold blue]")
        
        config = llm_client.config
        all_success = True
        tested_providers = set()
        
        # Test each workflow
        for workflow_name, workflow in config.workflows.items():
            provider_model = f"{workflow.provider}/{workflow.model}"
            
            if not silent:
                self.console.print(f"\n[dim]Testing {workflow_name.replace('_', ' ').title()}: {provider_model}...[/dim]")
            
            try:
                success, error_msg = llm_client.test_provider(workflow.provider, workflow.model)
                
                if success:
                    if not silent:
                        self.console.print(f"[green]✅ {workflow_name.replace('_', ' ').title()}: Connection successful[/green]")
                else:
                    if not silent:
                        self.console.print(f"[red]❌ {workflow_name.replace('_', ' ').title()}: Connection failed[/red]")
                    all_success = False
                    
                    # Show helpful tips (only once per provider)
                    if workflow.provider not in tested_providers:
                        self.show_connection_help(workflow.provider, workflow.model, llm_client)
                        tested_providers.add(workflow.provider)
                        
            except Exception as e:
                if not silent:
                    self.console.print(f"[red]❌ {workflow_name.replace('_', ' ').title()}: Error - {e}[/red]")
                all_success = False
        
        # Test fallback configurations if present
        fallback_tested = False
        for workflow_name, workflow in config.workflows.items():
            if workflow.fallback and not fallback_tested:
                if not silent:
                    self.console.print(f"\n[dim]Testing fallback: {workflow.fallback['provider']}/{workflow.fallback['model']}...[/dim]")
                
                try:
                    success, error_msg = llm_client.test_provider(workflow.fallback['provider'], workflow.fallback['model'])
                    if success:
                        if not silent:
                            self.console.print("[green]✅ Fallback: Connection successful[/green]")
                    else:
                        if not silent:
                            self.console.print("[red]❌ Fallback: Connection failed[/red]")
                except Exception as e:
                    if not silent:
                        self.console.print(f"[red]❌ Fallback: Error - {e}[/red]")
                
                fallback_tested = True
        
        if not silent:
            if all_success:
                self.console.print("\n[green bold]🎉 All configurations tested successfully![/green bold]")
            else:
                self.console.print("\n[yellow bold]⚠️  Some configurations failed - check the details above[/yellow bold]")
            
            self.console.print("\n[dim]Press Enter to continue...[/dim]")
            self.safe_input()
        
        return all_success

    def validate_configuration(self, llm_client: 'LLMClient', silent: bool = False) -> bool:
        """Validate the current configuration."""
        if not silent:
            self.console.print("\n[bold blue]🔍 Validating Configuration[/bold blue]")
        
        config = llm_client.config
        all_valid = True
        issues = []
        
        # Check that all required workflows are configured
        required_workflows = ["processing", "conversation", "jira_matching"]
        for workflow_name in required_workflows:
            if workflow_name not in config.workflows:
                issues.append(f"Missing workflow: {workflow_name}")
                all_valid = False
            else:
                workflow = config.workflows[workflow_name]
                
                # Check provider exists
                if workflow.provider not in config.providers:
                    issues.append(f"{workflow_name}: Provider '{workflow.provider}' not configured")
                    all_valid = False
                
                # Check model name is reasonable
                if not workflow.model or len(workflow.model.strip()) == 0:
                    issues.append(f"{workflow_name}: Model name is empty")
                    all_valid = False
        
        # Check provider configurations
        for provider_name, provider_config in config.providers.items():
            # Check API base is valid URL
            if not provider_config.api_base or not provider_config.api_base.startswith(('http://', 'https://')):
                issues.append(f"Provider '{provider_name}': Invalid API endpoint")
                all_valid = False
            
            # Check authentication is configured
            if not provider_config.api_key and not provider_config.auth_env:
                if provider_name not in ['lmstudio', 'ollama']:  # These don't need auth
                    issues.append(f"Provider '{provider_name}': No authentication configured")
                    all_valid = False
        
        # Check preset consistency
        if config.current_preset:
            if config.current_preset not in config.presets:
                issues.append(f"Current preset '{config.current_preset}' does not exist")
                all_valid = False
        
        if not silent:
            if all_valid:
                self.console.print("[green]✅ Configuration is valid[/green]")
            else:
                self.console.print("[yellow]⚠️  Configuration issues found:[/yellow]")
                for issue in issues:
                    self.console.print(f"   • {issue}")
        
        return all_valid

    def show_connection_help(self, provider: str, model: str, llm_client: 'LLMClient'):
        """Show provider-specific connection help."""
        if provider == "ollama":
            self.console.print("[yellow]💡 Ollama troubleshooting:[/yellow]")
            self.console.print(f"   • Make sure Ollama is running: `ollama serve`")
            self.console.print(f"   • Install the model: `ollama pull {model}`")
            self.console.print(f"   • Check available models: `ollama list`")
        elif provider == "lmstudio":
            self.console.print("[yellow]💡 LM Studio troubleshooting:[/yellow]")
            self.console.print(f"   • Make sure LM Studio is running with server enabled")
            self.console.print(f"   • Load a model in LM Studio")
            self.console.print(f"   • Check server is on http://localhost:1234")
        elif provider in ["openai", "anthropic", "gocode"]:
            provider_config = llm_client.config.providers.get(provider)
            if provider_config and provider_config.auth_env:
                self.console.print(f"[yellow]💡 {provider.title()} troubleshooting:[/yellow]")
                self.console.print(f"   • Make sure {provider_config.auth_env} is set in your environment")
                self.console.print(f"   • Check your API key is valid and has credits")
                self.console.print(f"   • Verify model '{model}' is available")
        else:
            self.console.print(f"[yellow]💡 {provider.title()} troubleshooting:[/yellow]")
            self.console.print(f"   • Check provider configuration and authentication")
            self.console.print(f"   • Verify endpoint is reachable")
            self.console.print(f"   • Confirm model '{model}' exists")
    
    def quit(self):
        """Quit the application."""
        self.console.print("\n[dim]Thanks for using Work Journal! 🎯[/dim]")
        self.running = False


def run_tui():
    """Run the TUI application."""
    tui = WorkJournalTUI()
    tui.run()


if __name__ == "__main__":
    run_tui()