import torch
import torch.nn as nn
import torch.nn.functional as F

from .convg import GraphConvolution


class CCGBlock(nn.Module):
    """
    Co-Crystal Graph Block: Core message-passing layer with global state integration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_edge_types: int,
        global_state_dim: int,
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.in_features = in_features
        self.out_features = out_features
        self.global_state_dim = global_state_dim

        self.global_fc = nn.Linear(global_state_dim // 2, out_features)

        self.graph_conv = GraphConvolution(
            in_features + out_features,  # node features + broadcasted global state
            out_features,
            num_edge_types,
        )

        self.bn_nodes = nn.BatchNorm1d(out_features)
        self.bn_global = nn.BatchNorm1d(out_features)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        global_state: torch.Tensor,
        subgraph_sizes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = x.shape

        global_state_split = global_state.view(
            batch_size, 2, -1
        )  # [batch, 2, global_dim/2]
        global_transformed = self.global_fc(
            global_state_split
        )  # [batch, 2, out_features]

        broadcasted_global = torch.zeros(
            batch_size, num_nodes, self.out_features, device=x.device
        )
        for i in range(batch_size):
            n1, n2 = subgraph_sizes[i]
            broadcasted_global[i, :n1] = global_transformed[i, 0]
            broadcasted_global[i, n1 : n1 + n2] = global_transformed[i, 1]

        x_concat = torch.cat([x, broadcasted_global], dim=-1)

        x_out = self.graph_conv(x_concat, adj)

        x_out_reshaped = x_out.view(-1, self.out_features)
        x_out_normalized = self.bn_nodes(x_out_reshaped)
        x_out = x_out_normalized.view(batch_size, num_nodes, self.out_features)

        if mask is not None:
            x_out = x_out * mask

        x_out = F.relu(x_out)

        global_transformed_flat = global_transformed.view(-1, self.out_features)
        global_normalized = self.bn_global(global_transformed_flat)
        global_normalized = global_normalized.view(batch_size, 2, self.out_features)
        global_normalized = F.relu(global_normalized)
        global_out = global_normalized.view(batch_size, -1)

        return x_out, global_out
