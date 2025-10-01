import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from .device import auto_select_device, get_available_devices
from .interactive import interactive_setup
from .pipeline import run_pipeline

app = typer.Typer(
    name="embedder",
    help="Text embedding and dimensionality reduction tool",
    rich_markup_mode="rich",
)

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@app.command()
def run(
    input_file: Path = typer.Argument(..., help="Input text file (one text per line)"),
    method: str = typer.Option(
        "umap", "--method", "-m", help="Reduction method: umap or autoencoder"
    ),
    model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--model",
        help="Sentence transformer model",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", "-d", help="Device: cpu, cuda:0, mps, auto"
    ),
    batch_size: int = typer.Option(
        256, "--batch-size", "-b", help="Embedding batch size"
    ),
    output_dir: Path = typer.Option(
        Path("./output"), "--output-dir", "-o", help="Output directory"
    ),
    save_prefix: str = typer.Option(
        "results", "--prefix", "-p", help="Save file prefix"
    ),
    n_components: int = typer.Option(
        3, "--components", "-c", help="Number of output dimensions"
    ),
    n_neighbors: int = typer.Option(15, "--neighbors", help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.1, "--min-dist", help="UMAP min_dist"),
    epochs: int = typer.Option(20, "--epochs", help="Autoencoder epochs"),
    hidden_dims: Optional[str] = typer.Option(
        "256,128", "--hidden", help="Autoencoder hidden dims (comma-sep)"
    ),
    lr: float = typer.Option(0.001, "--lr", help="Autoencoder learning rate"),
    sample_factor: Optional[float] = typer.Option(
        None, "--sample-factor", help="Randomly sample fraction of data (0.0-1.0)"
    ),
    random_seed: int = typer.Option(42, "--seed", help="Random seed for sampling"),
    no_viz: bool = typer.Option(False, "--no-viz", help="Skip visualization"),
    color_by_length: bool = typer.Option(
        False, "--color-length", help="Color by text length"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    setup_logging(verbose)

    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"🚀 [bold]Embedder v0.2.0[/bold] 🚀\nProcessing: {input_file}",
            style="blue",
        )
    )

    if device is None or device == "auto":
        device = auto_select_device()

    console.print(f"📱 Device: [green]{device}[/green]")
    console.print(f"🤖 Model: [cyan]{model}[/cyan]")
    console.print(f"⚙️  Method: [yellow]{method}[/yellow]")

    with open(input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

    console.print(f"📝 Loaded [bold]{len(texts)}[/bold] texts")

    if sample_factor is not None:
        if sample_factor <= 0 or sample_factor > 1:
            console.print(
                f"[red]Error: sample-factor must be between 0 and 1, got {sample_factor}[/red]"
            )
            raise typer.Exit(1)
        console.print(
            f"🎲 Sampling [cyan]{sample_factor*100:.1f}%[/cyan] of data (seed: {random_seed})"
        )

    umap_params = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": "cosine",
    }

    ae_params = {
        "latent_dim": n_components,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }

    if hidden_dims:
        ae_params["hidden_dims"] = [int(x.strip()) for x in hidden_dims.split(",")]

    viz_params = {"color_by_length": color_by_length}

    start_time = time.time()

    rows, df, embeddings, embed_3d = run_pipeline(
        texts=texts,
        method=method,
        hf_model=model,
        embed_batch_size=batch_size,
        device=device,
        umap_params=umap_params,
        ae_params=ae_params,
        save_prefix=save_prefix,
        output_dir=output_dir,
        visualize=not no_viz,
        viz_params=viz_params,
        sample_factor=sample_factor,
        random_seed=random_seed,
    )

    duration = time.time() - start_time
    console.print(f"\n✅ [green]Completed in {duration:.2f}s[/green]")
    console.print(f"📊 Output: {output_dir}")


@app.command()
def interactive():
    setup_logging(False)
    config = interactive_setup()

    with open(config["input_file"], "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

    start_time = time.time()

    rows, df, embeddings, embed_3d = run_pipeline(
        texts=texts,
        method=config["method"],
        hf_model=config["model"],
        embed_batch_size=config["embed_batch_size"],
        device=config["device"],
        umap_params=config["method_params"] if config["method"] == "umap" else None,
        ae_params=(
            config["method_params"] if config["method"] == "autoencoder" else None
        ),
        save_prefix=config["save_prefix"],
        output_dir=config["output_dir"],
        visualize=config["visualize"],
        viz_params=config["viz_params"],
        sample_factor=config.get("sample_factor"),
        random_seed=config.get("random_seed", 42),
    )

    duration = time.time() - start_time
    console.print(f"\n✅ [green]Completed in {duration:.2f}s[/green]")


@app.command()
def devices():
    devices = get_available_devices()
    console.print("Available devices:")
    for device in devices:
        console.print(f"  • {device}")


if __name__ == "__main__":
    app()
