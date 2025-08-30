#!/usr/bin/env python3
"""
DKMS CLI Client - Command line interface for the Distributed Key Management Service.

A Typer-based CLI tool for interacting with the DKMS REST API.
"""

import os
import sys
from typing import Optional
import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="DKMS CLI - Distributed Key Management Service Client")
console = Console()

# Configuration
DEFAULT_BASE_URL = "http://127.0.0.1:16724"
DEFAULT_API_KEY = "default-dev-key"

class DKMSClient:
    """HTTP client for DKMS API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def get_key(self, name: str) -> dict:
        """Retrieve a key from DKMS."""
        with httpx.Client() as client:
            response = client.get(f"{self.base_url}/key/{name}", headers=self.headers)
            response.raise_for_status()
            return response.json()

    def store_key(self, name: str, value: str) -> dict:
        """Store a key in DKMS."""
        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/key/{name}",
                headers=self.headers,
                json={"value": value}
            )
            response.raise_for_status()
            return response.json()

    def delete_key(self, name: str) -> dict:
        """Delete a key from DKMS."""
        with httpx.Client() as client:
            response = client.delete(f"{self.base_url}/key/{name}", headers=self.headers)
            response.raise_for_status()
            return response.json()

    def list_keys(self) -> dict:
        """List all keys in DKMS."""
        with httpx.Client() as client:
            response = client.get(f"{self.base_url}/keys", headers=self.headers)
            response.raise_for_status()
            return response.json()

def get_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> DKMSClient:
    """Create DKMS client with configuration from environment or defaults."""
    url = base_url or os.getenv("DKMS_URL", DEFAULT_BASE_URL)
    key = api_key or os.getenv("API_KEY", DEFAULT_API_KEY)
    return DKMSClient(url, key)

@app.command("get")
def get_key(
    name: str = typer.Argument(..., help="Name of the key to retrieve"),
    base_url: Optional[str] = typer.Option(None, "--url", "-u", help="DKMS server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for authentication"),
    show_value: bool = typer.Option(True, "--show-value/--hide-value", help="Show or hide the key value")
):
    """Retrieve a key from DKMS."""
    try:
        client = get_client(base_url, api_key)
        result = client.get_key(name)

        if show_value:
            rprint(f"[green]✓[/green] Retrieved key '[bold]{name}[/bold]'")
            rprint(f"Value: [yellow]{result['value']}[/yellow]")
        else:
            rprint(f"[green]✓[/green] Key '[bold]{name}[/bold]' exists")
            rprint("Value: [dim]<hidden>[/dim]")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            rprint(f"[red]✗[/red] Key '[bold]{name}[/bold]' not found")
        elif e.response.status_code == 401:
            rprint(f"[red]✗[/red] Authentication failed. Check your API key.")
        else:
            rprint(f"[red]✗[/red] Error: {e.response.status_code} - {e.response.text}")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]✗[/red] Failed to retrieve key: {e}")
        raise typer.Exit(1)

@app.command("set")
def store_key(
    name: str = typer.Argument(..., help="Name of the key to store"),
    value: Optional[str] = typer.Option(None, "--value", "-v", help="Value to store"),
    base_url: Optional[str] = typer.Option(None, "--url", "-u", help="DKMS server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for authentication"),
    prompt_value: bool = typer.Option(False, "--prompt", "-p", help="Prompt for value securely")
):
    """Store a key in DKMS."""
    # Get the value from different sources
    if prompt_value:
        value = typer.prompt("Enter the key value", hide_input=True)
    elif value is None:
        # Try to read from stdin
        if not sys.stdin.isatty():
            value = sys.stdin.read().strip()
        else:
            rprint("[red]✗[/red] No value provided. Use --value, --prompt, or pipe the value via stdin.")
            raise typer.Exit(1)

    try:
        client = get_client(base_url, api_key)
        result = client.store_key(name, value)
        rprint(f"[green]✓[/green] Stored key '[bold]{name}[/bold]' successfully")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            rprint(f"[red]✗[/red] Authentication failed. Check your API key.")
        else:
            rprint(f"[red]✗[/red] Error: {e.response.status_code} - {e.response.text}")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]✗[/red] Failed to store key: {e}")
        raise typer.Exit(1)

@app.command("delete")
def delete_key(
    name: str = typer.Argument(..., help="Name of the key to delete"),
    base_url: Optional[str] = typer.Option(None, "--url", "-u", help="DKMS server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for authentication"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt")
):
    """Delete a key from DKMS."""
    if not confirm:
        confirm_delete = typer.confirm(f"Are you sure you want to delete key '{name}'?")
        if not confirm_delete:
            rprint("Operation cancelled.")
            return

    try:
        client = get_client(base_url, api_key)
        result = client.delete_key(name)
        rprint(f"[green]✓[/green] {result['message']}")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            rprint(f"[red]✗[/red] Key '[bold]{name}[/bold]' not found")
        elif e.response.status_code == 401:
            rprint(f"[red]✗[/red] Authentication failed. Check your API key.")
        else:
            rprint(f"[red]✗[/red] Error: {e.response.status_code} - {e.response.text}")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]✗[/red] Failed to delete key: {e}")
        raise typer.Exit(1)

@app.command("list")
def list_keys(
    base_url: Optional[str] = typer.Option(None, "--url", "-u", help="DKMS server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for authentication"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, simple")
):
    """List all keys in DKMS."""
    try:
        client = get_client(base_url, api_key)
        result = client.list_keys()
        keys = result['keys']

        if not keys:
            rprint("[yellow]No keys found[/yellow]")
            return

        if format == "json":
            import json
            print(json.dumps(result, indent=2))
        elif format == "simple":
            for key in keys:
                print(key)
        else:  # table format (default)
            table = Table(title="DKMS Keys")
            table.add_column("Key Name", style="cyan", no_wrap=True)
            table.add_column("Index", style="magenta")

            for i, key in enumerate(keys, 1):
                table.add_row(key, str(i))

            console.print(table)
            rprint(f"\n[green]Total: {len(keys)} keys[/green]")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            rprint(f"[red]✗[/red] Authentication failed. Check your API key.")
        else:
            rprint(f"[red]✗[/red] Error: {e.response.status_code} - {e.response.text}")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]✗[/red] Failed to list keys: {e}")
        raise typer.Exit(1)

@app.command("config")
def show_config(
    base_url: Optional[str] = typer.Option(None, "--url", "-u", help="DKMS server URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for authentication")
):
    """Show current configuration."""
    url = base_url or os.getenv("DKMS_URL", DEFAULT_BASE_URL)
    key = api_key or os.getenv("API_KEY", DEFAULT_API_KEY)

    table = Table(title="DKMS CLI Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Source", style="green")

    # Base URL
    url_source = "argument" if base_url else ("environment" if os.getenv("DKMS_URL") else "default")
    table.add_row("Base URL", url, url_source)

    # API Key (masked for security)
    key_masked = f"{key[:8]}..." if len(key) > 8 else "***"
    key_source = "argument" if api_key else ("environment" if os.getenv("API_KEY") else "default")
    table.add_row("API Key", key_masked, key_source)

    console.print(table)

    rprint("\n[bold]Environment Variables:[/bold]")
    rprint("• DKMS_URL - DKMS server URL")
    rprint("• API_KEY - Authentication API key")

if __name__ == "__main__":
    app()
