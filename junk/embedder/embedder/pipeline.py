import numpy as np
import pandas as pd
import pickle
import torch
import random
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from sklearn.manifold import trustworthiness

from .embeddings import compute_embeddings
from .reduction import umap_reduce, train_autoencoder
from .visualization import visualize_embeddings

logger = logging.getLogger(__name__)

def make_results(texts: List[str], embeddings_nd: np.ndarray, embeddings_3d: np.ndarray):
    rows = []
    for t, nd, d3 in zip(texts, embeddings_nd, embeddings_3d):
        rows.append({
            "text": t,
            "embedding_nd": nd.astype(float).tolist(),
            "embedding_3d": d3.astype(float).tolist()
        })
    df = pd.DataFrame(rows)
    return rows, df

def sample_texts(texts: List[str], sample_factor: Optional[float] = None, random_seed: int = 42) -> Tuple[List[str], List[int]]:
    """Sample a subset of texts randomly."""
    if sample_factor is None or sample_factor >= 1.0:
        return texts, list(range(len(texts)))
    
    if sample_factor <= 0.0:
        raise ValueError("sample_factor must be between 0 and 1")
    
    random.seed(random_seed)
    n_samples = int(len(texts) * sample_factor)
    n_samples = max(1, n_samples)  # Ensure at least 1 sample
    
    original_indices = list(range(len(texts)))
    sampled_indices = random.sample(original_indices, n_samples)
    sampled_indices.sort()  # Keep original order
    
    sampled_texts = [texts[i] for i in sampled_indices]
    
    return sampled_texts, sampled_indices

def run_pipeline(
    texts: List[str],
    method: str = "umap",
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size: int = 256,
    device: Optional[str] = None,
    umap_params: Optional[Dict[str, Any]] = None,
    ae_params: Optional[Dict[str, Any]] = None,
    save_prefix: Optional[str] = "results",
    output_dir: Path = Path("./output"),
    visualize: bool = True,
    viz_params: Optional[Dict[str, Any]] = None,
    sample_factor: Optional[float] = None,
    random_seed: int = 42
) -> Tuple[List[dict], pd.DataFrame, np.ndarray, np.ndarray]:
    
    if umap_params is None:
        umap_params = {"n_components": 3, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}
    if ae_params is None:
        ae_params = {"latent_dim": 3, "hidden_dims": [512, 256], "batch_size": 1024, "epochs": 30, "lr": 1e-3}
    if viz_params is None:
        viz_params = {"color_by_length": False, "opacity": 0.7}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample texts if requested
    original_count = len(texts)
    if sample_factor is not None and sample_factor < 1.0:
        texts, sampled_indices = sample_texts(texts, sample_factor, random_seed)
        logger.info(f"Sampled {len(texts)} texts from {original_count} ({sample_factor*100:.1f}%)")
    else:
        sampled_indices = list(range(len(texts)))

    logger.info(f"Starting pipeline with {len(texts)} texts using {method}")
    
    embeddings = compute_embeddings(
        texts, 
        model_name=hf_model, 
        batch_size=embed_batch_size, 
        device=device
    )

    if method == "umap":
        logger.info("Running UMAP reduction")
        embed_3d = umap_reduce(embeddings, **umap_params)
        method_tag = "umap"
    elif method == "autoencoder":
        logger.info("Training autoencoder")
        model, embed_3d = train_autoencoder(
            embeddings,
            device=device,
            **ae_params
        )
        method_tag = "autoencoder"
        if save_prefix:
            torch.save(model.state_dict(), output_dir / f"{save_prefix}_ae_state.pt")
            logger.info(f"Saved model to {output_dir / f'{save_prefix}_ae_state.pt'}")
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'umap' or 'autoencoder'")

    try:
        trust = trustworthiness(embeddings, embed_3d, n_neighbors=12)
        logger.info(f"Trustworthiness (k=12): {trust:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute trustworthiness: {e}")
        trust = None

    rows, df = make_results(texts, embeddings, embed_3d)
    
    # Add sampling information to results
    if sample_factor is not None and sample_factor < 1.0:
        for i, row in enumerate(rows):
            row['original_index'] = sampled_indices[i]
        df['original_index'] = sampled_indices

    if save_prefix:
        parquet_file = output_dir / f"{save_prefix}_{method_tag}.parquet"
        pickle_file = output_dir / f"{save_prefix}_{method_tag}.pkl"
        
        df.to_parquet(parquet_file, index=False)
        with open(pickle_file, "wb") as f:
            pickle.dump(rows, f)
        
        # Save sampling metadata if used
        if sample_factor is not None and sample_factor < 1.0:
            metadata_file = output_dir / f"{save_prefix}_{method_tag}_metadata.json"
            import json
            metadata = {
                "original_count": original_count,
                "sampled_count": len(texts),
                "sample_factor": sample_factor,
                "random_seed": random_seed,
                "sampled_indices": sampled_indices
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved sampling metadata to {metadata_file}")
        
        logger.info(f"Saved results to {parquet_file} and {pickle_file}")

    if visualize:
        html_file = output_dir / f"{save_prefix}_{method_tag}_viz.html"
        visualize_embeddings(texts, embed_3d, save_html=str(html_file), **viz_params)

    return rows, df, embeddings, embed_3d