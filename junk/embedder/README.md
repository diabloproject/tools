# Embedder

A comprehensive tool for text embedding and dimensionality reduction with rich CLI interface.

## Features

- 🚀 **Modern CLI** with Rich formatting and Typer
- 📱 **Device Selection** - Auto-detect and choose between CPU, CUDA, MPS
- 🤖 **Multiple Models** - Support for various Sentence Transformers
- 🔄 **Reduction Methods** - UMAP and Autoencoder
- 📊 **Visualization** - Interactive 2D/3D plots with Plotly
- 💾 **Multiple Formats** - Save as Parquet, Pickle
- 🎯 **Interactive Mode** - Guided setup with prompts
- 📝 **Comprehensive Logging** - Rich formatted logs
- 🎲 **Random Sampling** - Process subset of data for testing/exploration

## Installation

```bash
# Install dependencies
uv add typer rich torch scikit-learn tqdm numpy pyarrow pandas plotly sentence-transformers umap-learn
```

## Usage

### Command Line Interface

```bash
# Basic usage with UMAP
uv run python -m embedder run input.txt

# Use autoencoder with custom parameters
uv run python -m embedder run input.txt --method autoencoder --epochs 30 --components 3

# Specify device and visualization options  
uv run python -m embedder run input.txt --device mps --color-length --components 2

# Random sampling for large datasets
uv run python -m embedder run large_dataset.txt --sample-factor 0.1 --seed 42

# Full parameter example
uv run python -m embedder run input.txt \
  --method umap \
  --model "sentence-transformers/all-mpnet-base-v2" \
  --device cuda:0 \
  --components 3 \
  --neighbors 20 \
  --min-dist 0.05 \
  --batch-size 512 \
  --sample-factor 0.5 \
  --seed 123 \
  --output-dir ./results \
  --prefix experiment_1 \
  --color-length \
  --verbose
```

### Interactive Mode

```bash
uv run python -m embedder interactive
```

This launches a guided setup with prompts for all parameters.

### Available Commands

```bash
# Show help
uv run python -m embedder --help
uv run python -m embedder run --help

# List available devices
uv run python -m embedder devices

# Interactive setup
uv run python -m embedder interactive
```

## Input Format

Create a text file with one text sample per line:

```
The quick brown fox jumps over the lazy dog
Machine learning algorithms can identify patterns
Natural language processing enables computers to understand text
```

## Parameters

### Embedding Parameters
- `--model`: Sentence Transformer model name
- `--device`: Computing device (cpu, cuda:0, mps, auto)
- `--batch-size`: Embedding batch size

### UMAP Parameters  
- `--components`: Output dimensions (default: 3)
- `--neighbors`: Number of neighbors (default: 15)
- `--min-dist`: Minimum distance (default: 0.1)

### Autoencoder Parameters
- `--components`: Latent dimensions (default: 3)  
- `--epochs`: Training epochs (default: 20)
- `--hidden`: Hidden layer sizes, comma-separated (default: "256,128")
- `--lr`: Learning rate (default: 0.001)

### Output Parameters
- `--output-dir`: Output directory (default: ./output)
- `--prefix`: File name prefix (default: results)
- `--no-viz`: Skip visualization
- `--color-length`: Color points by text length

### Sampling Parameters
- `--sample-factor`: Random sampling fraction (0.0-1.0, e.g., 0.5 = 50%)
- `--seed`: Random seed for reproducible sampling (default: 42)

## Output Files

The tool generates several output files:

- `{prefix}_{method}.parquet` - Results in Parquet format
- `{prefix}_{method}.pkl` - Results in Pickle format  
- `{prefix}_{method}_viz.html` - Interactive visualization
- `{prefix}_{method}_metadata.json` - Sampling metadata (if sampling used)
- `{prefix}_ae_state.pt` - Autoencoder model weights (if using autoencoder)

## Examples

### Quick Start

```bash
# Create sample data
echo -e "Hello world\nMachine learning\nDeep learning\nNatural language" > sample.txt

# Run with UMAP on full dataset
uv run python -m embedder run sample.txt --components 2

# Run with 50% random sampling
uv run python -m embedder run sample.txt --sample-factor 0.5 --seed 123

# Run with autoencoder on sample
uv run python -m embedder run sample.txt --method autoencoder --epochs 10 --sample-factor 0.3
```

### Sampling Examples

```bash
# Process 10% of a large dataset for quick testing
uv run python -m embedder run large_dataset.txt --sample-factor 0.1

# Reproducible sampling with fixed seed
uv run python -m embedder run dataset.txt --sample-factor 0.25 --seed 42

# Different random samples from same dataset
uv run python -m embedder run dataset.txt --sample-factor 0.5 --seed 123 --prefix sample_a
uv run python -m embedder run dataset.txt --sample-factor 0.5 --seed 456 --prefix sample_b
```

### Advanced Usage

```bash
# High-quality embeddings with large model
uv run python -m embedder run large_dataset.txt \
  --model "sentence-transformers/all-mpnet-base-v2" \
  --method umap \
  --neighbors 50 \
  --min-dist 0.01 \
  --batch-size 1024 \
  --device cuda:0 \
  --verbose

# Custom autoencoder architecture
uv run python -m embedder run dataset.txt \
  --method autoencoder \
  --hidden "512,256,128" \
  --epochs 50 \
  --lr 0.0005 \
  --batch-size 256
```

## Project Structure

```
embedder/
├── __init__.py
├── __main__.py      # Main CLI entry point
├── device.py        # Device detection and selection
├── embeddings.py    # Text embedding computation
├── reduction.py     # UMAP and autoencoder implementations
├── visualization.py # Plotting and visualization
├── pipeline.py      # Main processing pipeline
└── interactive.py   # Interactive UI components
```

## Development

The project is structured as a Python package that can be run with:

```bash
uv run python -m embedder [command] [options]
```

All dependencies are managed with `uv` and the project follows modern Python packaging standards.