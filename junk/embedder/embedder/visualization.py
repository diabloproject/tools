import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List
import logging

logger = logging.getLogger(__name__)

def visualize_embeddings(
    texts: List[str],
    coords,
    save_html: str = "embedding_viz.html",
    color_by_length: bool = False,
    opacity: float = 0.7,
    marker_size: int = 3
):
    n_dims = coords.shape[1]
    logger.info(f"Creating {n_dims}D visualization with {len(texts)} points")
    
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "text": texts,
        "length": [len(t) for t in texts]
    })
    
    if n_dims >= 3:
        df["z"] = coords[:, 2]
    
    color = "length" if color_by_length else None
    
    if n_dims >= 3:
        fig = px.scatter_3d(
            df,
            x="x", y="y", z="z",
            color=color,
            hover_name="text",
            opacity=opacity,
            size_max=5
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
                zaxis_title="Dim 3"
            )
        )
    else:
        fig = px.scatter(
            df,
            x="x", y="y",
            color=color,
            hover_name="text",
            opacity=opacity,
            size_max=5
        )
        fig.update_layout(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2"
        )
    
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    fig.write_html(save_html)
    logger.info(f"Visualization saved to {save_html}")
    return fig

def visualize_3d(texts, coords_3d, **kwargs):
    return visualize_embeddings(texts, coords_3d, **kwargs)