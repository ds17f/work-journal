"""Main CLI interface for work-journal."""

import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm

from .llm import LLMClient
from .storage import Storage
from .date_parser import DateParser
from .entry_processor import EntryProcessor

console = Console()
app = typer.Typer(help="A CLI tool for tracking work accomplishments.")


@app.command()
def hello():
    """Test command to verify the CLI is working."""
    console.print(Panel("Hello, Dude! I like your style.", title="Work Journal"))


@app.command()
def add(
    date: str = typer.Option("today", help="Date for the entry (today, yesterday, or YYYY-MM-DD)"),
    text: Optional[str] = typer.Option(None, help="Entry text (if not provided, will prompt interactively)")
):
    """Add a new journal entry."""
    try:
        # Parse the date
        parsed_date = DateParser.parse_date(date)
        display_date = DateParser.format_date_display(parsed_date)
        
        console.print(Panel(f"Creating journal entry for [bold]{display_date}[/bold] ({parsed_date})", style="bold green"))
        
        # Get input from user
        if text:
            raw_input = text
            console.print(f"[dim]Processing provided text: {raw_input}[/dim]")
        else:
            console.print("\n[bold cyan]What did you accomplish?[/bold cyan]")
            try:
                raw_input = Prompt.ask("Tell me about your work")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Entry cancelled.[/yellow]")
                return
        
        if not raw_input.strip():
            console.print("[yellow]No input provided. Entry cancelled.[/yellow]")
            return
        
        console.print(f"\n[bold]Raw input:[/bold] [dim]{raw_input}[/dim]")
        console.print(f"[bold]Parsed date:[/bold] [dim]{parsed_date}[/dim]")
        
        # Process entry with LLM
        try:
            from .llm import LLMClient
            llm_client = LLMClient()
            current_model = llm_client.config.workflows["processing"]
            model_info = f"{current_model.provider}/{current_model.model}"
        except:
            model_info = "LLM"
        
        console.print(f"\n[cyan]🤖 Processing with {model_info}...[/cyan]")
        
        processor = EntryProcessor()
        entry = processor.process_entry(raw_input, parsed_date)
        
        # Display structured results
        console.print("\n[bold green]✨ Structured Entry:[/bold green]")
        
        # Show processed data in a nice format
        processed = entry.processed
        
        console.print(f"\n[bold]Summary:[/bold] {processed.summary}")
        console.print(f"[bold]Work:[/bold] {processed.work}")
        
        if processed.collaborators:
            console.print(f"[bold]Collaborators:[/bold] {', '.join(processed.collaborators)}")
        
        console.print(f"[bold]Impact Scope:[/bold] {processed.impact.scope}")
        if processed.impact.metrics:
            console.print(f"[bold]Metrics:[/bold] {'; '.join(processed.impact.metrics)}")
        console.print(f"[bold]Business Value:[/bold] {processed.impact.business_value}")
        
        if processed.technical_details:
            console.print(f"[bold]Technical Details:[/bold] {'; '.join(processed.technical_details)}")
        
        console.print(f"[bold]Tags:[/bold] {', '.join(processed.tags)}")
        
        # Save to storage
        console.print("\n[cyan]💾 Saving to storage...[/cyan]")
        storage = Storage()
        storage.save_entry(entry)
        
        console.print(f"✅ Entry saved for {display_date}!")
        
        # Ask for additional entries
        if not text:  # Only ask if we were in interactive mode
            try:
                another = Confirm.ask("\nWant to add another entry for today?", default=False)
                if another:
                    # Recursively call add for the same date
                    add(date=parsed_date)
            except (EOFError, KeyboardInterrupt):
                pass
        
    except Exception as e:
        console.print(f"[red]Error creating entry: {e}[/red]")


config_app = typer.Typer(help="Configuration management for LLM providers and workflows")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    try:
        llm_client = LLMClient()
        config = llm_client.config
        
        console.print(Panel("Current Configuration", style="bold blue"))
        
        # Show current preset
        if config.current_preset:
            console.print(f"[bold]Active Preset:[/bold] {config.current_preset}")
        
        # Show workflows
        console.print("\n[bold]Workflow Configuration:[/bold]")
        workflow_table = Table()
        workflow_table.add_column("Workflow Step")
        workflow_table.add_column("Provider")
        workflow_table.add_column("Model")
        workflow_table.add_column("Fallback")
        
        for step, workflow in config.workflows.items():
            fallback_str = f"{workflow.fallback['provider']}:{workflow.fallback['model']}" if workflow.fallback else "None"
            workflow_table.add_row(step, workflow.provider, workflow.model, fallback_str)
        
        console.print(workflow_table)
        
        # Show available presets
        console.print(f"\n[bold]Available Presets:[/bold] {', '.join(config.presets.keys())}")
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")


@config_app.command("init")
def config_init():
    """Initialize configuration and .env file."""
    try:
        storage = Storage()
        env_file = storage.ensure_env_file()
        
        console.print(Panel("Configuration Initialization", style="bold green"))
        console.print(f"Created .env file at: [cyan]{env_file}[/cyan]")
        
        # Initialize LLM client (creates default config)
        llm_client = LLMClient()
        console.print("✅ Default configuration created")
        
        console.print(f"\n[yellow]Next steps:[/yellow]")
        console.print(f"1. Add your GOCODE_AUTH_TOKEN to: {env_file}")
        console.print(f"2. Run: [cyan]work-journal test-llm[/cyan] to verify connectivity")
        console.print(f"3. Use: [cyan]work-journal config show[/cyan] to view settings")
        
    except Exception as e:
        console.print(f"[red]Error initializing configuration: {e}[/red]")


@config_app.command("use-preset")
def config_use_preset(preset_name: str):
    """Switch to a different preset configuration."""
    try:
        llm_client = LLMClient()
        
        if preset_name not in llm_client.config.presets:
            console.print(f"[red]Preset '{preset_name}' not found.[/red]")
            console.print(f"Available presets: {', '.join(llm_client.config.presets.keys())}")
            return
        
        # Update workflows to match preset
        preset_config = llm_client.config.presets[preset_name]
        llm_client.config.workflows = preset_config
        llm_client.config.current_preset = preset_name
        
        # Save configuration
        llm_client.storage.save_config(llm_client.config)
        
        console.print(f"✅ Switched to preset: [bold]{preset_name}[/bold]")
        
    except Exception as e:
        console.print(f"[red]Error switching preset: {e}[/red]")


@app.command()
def test_llm(provider: Optional[str] = typer.Option(None, help="Test specific provider (gocode, lmstudio, ollama)")):
    """Test LLM connectivity for configured providers."""
    try:
        llm_client = LLMClient()
        
        if provider:
            # Test specific provider
            console.print(f"Testing provider: [bold]{provider}[/bold]")
            success, error_msg = llm_client.test_provider(provider)
            if success:
                console.print(f"✅ {provider} connection successful")
            else:
                console.print(f"❌ {provider} connection failed")
        else:
            # Test all configured providers
            console.print(Panel("Testing LLM Provider Connectivity", style="bold blue"))
            
            # Get unique providers from workflows
            providers_to_test = set()
            for workflow in llm_client.config.workflows.values():
                providers_to_test.add(workflow.provider)
                if workflow.fallback:
                    providers_to_test.add(workflow.fallback["provider"])
            
            results = {}
            for provider_name in providers_to_test:
                console.print(f"Testing [bold]{provider_name}[/bold]...")
                try:
                    success, error_msg = llm_client.test_provider(provider_name)
                    results[provider_name] = success
                    if success:
                        console.print(f"  ✅ Connected successfully")
                    else:
                        console.print(f"  ❌ Connection failed")
                except Exception as e:
                    results[provider_name] = False
                    console.print(f"  ❌ Error: {e}")
            
            # Summary
            successful = sum(1 for r in results.values() if r)
            total = len(results)
            console.print(f"\n[bold]Summary:[/bold] {successful}/{total} providers connected successfully")
            
    except Exception as e:
        console.print(f"[red]Error testing LLM providers: {e}[/red]")


@app.command()
def tui():
    """Launch the interactive Terminal User Interface."""
    try:
        from .tui import run_tui
        run_tui()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
    except Exception as e:
        console.print(f"[red]Error running TUI: {e}[/red]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()