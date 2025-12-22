import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for graph-level readout.
    Equivalent to multi_head_global_attention in the original implementation.

    NOTE: This is NOT the same as nn.MultiheadAttention!
    - Uses element-wise attention (not dot-product)
    - Applies segment-wise softmax (respects graph boundaries)
    - Pools to graph-level representation (not per-node)
    """

    def __init__(self, in_features: int, num_heads: int = 2) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.in_features = in_features
        self.num_heads = num_heads

        self.attention_transform = nn.Linear(in_features, num_heads * in_features)
        self.attention_weights = nn.Parameter(torch.Tensor(1, num_heads, in_features))

    def forward(self, x: torch.Tensor, graph_sizes: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x_transformed = self.attention_transform(
            x
        )  # [batch, num_nodes, num_heads * in_features]
        x_transformed = x_transformed.view(
            batch_size, -1, self.num_heads, self.in_features
        )

        attention_scores = (self.attention_weights * x_transformed).sum(
            dim=-1
        )  # [batch, num_nodes, num_heads]
        attention_scores = F.leaky_relu(attention_scores, negative_slope=0.2)

        attention_weights = torch.zeros_like(attention_scores)
        for i in range(batch_size):
            num_atoms = graph_sizes[i]
            attention_weights[i, :num_atoms] = F.softmax(
                attention_scores[i, :num_atoms], dim=0
            )

        attention_weights = attention_weights.unsqueeze(
            -1
        )  # [batch, num_nodes, num_heads, 1]
        x_transformed = x_transformed.permute(
            0, 2, 1, 3
        )  # [batch, num_heads, num_nodes, in_features]
        attention_weights = attention_weights.permute(
            0, 2, 1, 3
        )  # [batch, num_heads, num_nodes, 1]

        weighted = (
            x_transformed * attention_weights
        )  # [batch, num_heads, num_nodes, in_features]

        graph_embedding = weighted.sum(dim=2)  # [batch, num_heads, in_features]
        graph_embedding = graph_embedding.view(
            batch_size, -1
        )  # [batch, num_heads * in_features]

        return graph_embedding
