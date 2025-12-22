import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import CCGBlock
from .mha import MultiHeadAttention


class CCGNet(nn.Module):
    """
    Complete CCGNet model for cocrystal prediction.
    """

    def __init__(
        self,
        node_features: int = 34,
        num_edge_types: int = 4,
        global_state_dim: int = 24,
        num_classes: int = 2,
    ) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]

        self.node_features = node_features
        self.num_edge_types = num_edge_types
        self.global_state_dim = global_state_dim

        # Message passing blocks (alternating 64 and 16 filters)
        self.ccg_block1 = CCGBlock(node_features, 64, num_edge_types, global_state_dim)
        self.ccg_block2 = CCGBlock(64, 16, num_edge_types, 128)  # 64*2 for global state
        self.ccg_block3 = CCGBlock(16, 64, num_edge_types, 32)  # 16*2
        self.ccg_block4 = CCGBlock(64, 16, num_edge_types, 128)  # 64*2

        # Readout
        self.attention = MultiHeadAttention(16, num_heads=2)

        # Prediction head
        # Input: num_heads * feature_dim + final_global_state_dim = 2*16 + 32 = 64
        self.fc1 = nn.Linear(64, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.457)

        self.fc2 = nn.Linear(256, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.457)

        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.457)

        self.fc_final = nn.Linear(256, num_classes)

    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        global_state: torch.Tensor,
        subgraph_sizes: torch.Tensor,
        graph_sizes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, node_features]
            adj: [batch, num_nodes, num_edge_types, num_nodes]
            global_state: [batch, global_state_dim]
            subgraph_sizes: [batch, 2] - atoms per molecule
            graph_sizes: [batch] - total atoms per graph
            mask: [batch, num_nodes, 1] - node mask (optional)

        Returns:
            logits: [batch, num_classes]
        """
        # Message passing
        x, global_state = self.ccg_block1(
            node_features, adj, global_state, subgraph_sizes, mask
        )
        x, global_state = self.ccg_block2(x, adj, global_state, subgraph_sizes, mask)
        x, global_state = self.ccg_block3(x, adj, global_state, subgraph_sizes, mask)
        x, global_state = self.ccg_block4(x, adj, global_state, subgraph_sizes, mask)

        # Readout
        graph_embedding = self.attention(x, graph_sizes)

        # Concatenate with global state
        combined = torch.cat([graph_embedding, global_state], dim=-1)

        # Prediction head
        out = self.fc1(combined)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout3(out)

        logits = self.fc_final(out)

        return logits
