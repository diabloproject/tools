from typing import Dict, Any, List, Optional
import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from .device import get_available_devices, get_device_info

console = Console()

AVAILABLE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2", 
    "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
]

def select_device() -> str:
    devices = get_available_devices()
    
    if len(devices) == 1:
        return devices[0]
    
    table = Table(title="Available Devices")
    table.add_column("Index", style="cyan")
    table.add_column("Device", style="magenta")
    table.add_column("Info", style="green")
    
    for i, device in enumerate(devices):
        info = get_device_info(device)
        device_info = f"{info.get('name', 'Unknown')} ({info.get('memory_gb', 0):.1f}GB)" if 'memory_gb' in info else info.get('name', 'Unknown')
        table.add_row(str(i), device, device_info)
    
    console.print(table)
    
    choice = IntPrompt.ask("Select device", default=0, show_default=True)
    if 0 <= choice < len(devices):
        return devices[choice]
    return devices[0]

def select_model() -> str:
    table = Table(title="Available Models")
    table.add_column("Index", style="cyan")
    table.add_column("Model", style="magenta")
    
    for i, model in enumerate(AVAILABLE_MODELS):
        table.add_row(str(i), model)
    
    console.print(table)
    
    choice = IntPrompt.ask("Select model", default=0, show_default=True)
    if 0 <= choice < len(AVAILABLE_MODELS):
        return AVAILABLE_MODELS[choice]
    return AVAILABLE_MODELS[0]

def configure_method() -> tuple:
    methods = ["umap", "autoencoder"]
    
    table = Table(title="Reduction Methods")
    table.add_column("Index", style="cyan") 
    table.add_column("Method", style="magenta")
    table.add_column("Description", style="green")
    
    table.add_row("0", "umap", "Fast, good for exploration")
    table.add_row("1", "autoencoder", "Neural network, more customizable")
    
    console.print(table)
    
    choice = IntPrompt.ask("Select method", default=0, show_default=True)
    method = methods[choice] if 0 <= choice < len(methods) else methods[0]
    
    params = {}
    
    if method == "umap":
        console.print("\n[bold]UMAP Parameters[/bold]")
        params = {
            "n_components": IntPrompt.ask("Number of components", default=3),
            "n_neighbors": IntPrompt.ask("Number of neighbors", default=15),
            "min_dist": FloatPrompt.ask("Minimum distance", default=0.1),
            "metric": Prompt.ask("Distance metric", default="cosine")
        }
    else:
        console.print("\n[bold]Autoencoder Parameters[/bold]")
        params = {
            "latent_dim": IntPrompt.ask("Latent dimensions", default=3),
            "epochs": IntPrompt.ask("Training epochs", default=20),
            "batch_size": IntPrompt.ask("Batch size", default=512),
            "lr": FloatPrompt.ask("Learning rate", default=0.001)
        }
        
        hidden_str = Prompt.ask("Hidden layer sizes (comma-separated)", default="256,128")
        params["hidden_dims"] = [int(x.strip()) for x in hidden_str.split(",")]
    
    return method, params

def interactive_setup() -> Dict[str, Any]:
    console.print(Panel.fit("🚀 [bold]Embedder Interactive Setup[/bold] 🚀", style="blue"))
    
    input_file = Prompt.ask("Input text file path")
    if not Path(input_file).exists():
        console.print(f"[red]Error: File {input_file} does not exist[/red]")
        raise typer.Exit(1)
    
    device = select_device()
    model = select_model()
    method, method_params = configure_method()
    
    embed_batch_size = IntPrompt.ask("Embedding batch size", default=256)
    
    # Sampling options
    use_sampling = Confirm.ask("Use random sampling?", default=False)
    sample_factor = None
    random_seed = 42
    if use_sampling:
        sample_factor = FloatPrompt.ask("Sample factor (0.0-1.0)", default=0.5)
        if sample_factor <= 0 or sample_factor > 1:
            console.print("[red]Invalid sample factor, using 0.5[/red]")
            sample_factor = 0.5
        random_seed = IntPrompt.ask("Random seed", default=42)
    
    output_dir = Prompt.ask("Output directory", default="./output")
    save_prefix = Prompt.ask("Save prefix", default="results")
    
    visualize = Confirm.ask("Generate 3D visualization?", default=True)
    viz_params = {}
    if visualize:
        viz_params["color_by_length"] = Confirm.ask("Color by text length?", default=False)
    
    config = {
        "input_file": input_file,
        "device": device,
        "model": model,
        "method": method,
        "method_params": method_params,
        "embed_batch_size": embed_batch_size,
        "output_dir": output_dir,
        "save_prefix": save_prefix,
        "visualize": visualize,
        "viz_params": viz_params,
        "sample_factor": sample_factor,
        "random_seed": random_seed
    }
    
    console.print("\n[green]✓ Configuration complete![/green]")
    return config