import numpy as np
from typing import List, Tuple, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import umap
from tqdm import tqdm
import logging
from .device import validate_device

logger = logging.getLogger(__name__)

def umap_reduce(
    embeddings: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    verbose: bool = False
) -> np.ndarray:
    logger.info(f"Running UMAP reduction: {embeddings.shape} -> {n_components}D")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=verbose
    )
    
    embed_3d = reducer.fit_transform(embeddings)
    logger.info(f"UMAP completed: output shape {embed_3d.shape}")
    return embed_3d

class SimpleAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int = 3):
        super().__init__()
        dims = [input_dim] + hidden_dims
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)])
        enc_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        dims_rev = [latent_dim] + hidden_dims[::-1] + [input_dim]
        for i in range(len(dims_rev) - 1):
            dec_layers.append(nn.Linear(dims_rev[i], dims_rev[i+1]))
            if i < len(dims_rev) - 2:
                dec_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return z, xrec

def train_autoencoder(
    embeddings: np.ndarray,
    latent_dim: int = 3,
    hidden_dims: List[int] = None,
    batch_size: int = 512,
    epochs: int = 20,
    lr: float = 1e-3,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[SimpleAE, np.ndarray]:
    if hidden_dims is None:
        hidden_dims = [256, 128]
    
    device = validate_device(device)
    logger.info(f"Training autoencoder on {device}: {embeddings.shape} -> {latent_dim}D")

    X = torch.from_numpy(embeddings.astype(np.float32))
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = SimpleAE(
        input_dim=embeddings.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        model.train()
        for (batch,) in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", disable=not verbose):
            batch = batch.to(device)
            optimizer.zero_grad()
            z, xrec = model(batch)
            loss = criterion(xrec, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        
        epoch_loss /= len(dataset)
        if verbose:
            logger.info(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(embeddings.astype(np.float32)).to(device)
        z_all, _ = model(X_all)
        z_np = z_all.cpu().numpy()
    
    logger.info(f"Autoencoder training completed: output shape {z_np.shape}")
    return model, z_np