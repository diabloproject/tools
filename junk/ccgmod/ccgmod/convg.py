import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Graph convolution layer that processes multichannel adjacency tensors.
    """

    def __init__(
        self, in_features: int, out_features: int, num_edge_types: int
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.in_features = in_features
        self.out_features = out_features
        self.num_edge_types = num_edge_types

        self.W = nn.Parameter(torch.Tensor(in_features * num_edge_types, out_features))
        self.W_I = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        aggregated: list[torch.Tensor] = []
        for k in range(self.num_edge_types):
            # adj[:, :, k, :]: [batch, num_nodes, num_nodes]
            # x: [batch, num_nodes, in_features]
            agg_k = torch.matmul(adj[:, :, k, :], x)  # [batch, num_nodes, in_features]
            aggregated.append(agg_k)

        neighbor_agg = torch.cat(
            aggregated, dim=-1
        )  # [batch, num_nodes, in_features * num_edge_types]
        neighbor_out = torch.matmul(
            neighbor_agg, self.W
        )  # [batch, num_nodes, out_features]
        self_out = torch.matmul(x, self.W_I)  # [batch, num_nodes, out_features]

        return neighbor_out + self_out + self.bias
