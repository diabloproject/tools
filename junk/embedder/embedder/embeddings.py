import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .device import validate_device

logger = logging.getLogger(__name__)


def compute_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    device = validate_device(device)
    logger.info(f"Loading model {model_name} on {device}")

    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    logger.info(f"Computing embeddings for {len(texts)} texts")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings
