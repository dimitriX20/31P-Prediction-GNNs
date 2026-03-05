from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum


class EdgeModel(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, node_in_dim: int, hidden_dim: int):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        _, col = edge_index
        agg = scatter_sum(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, agg], dim=1)
        return self.node_mlp(out)


class GlobalModel(nn.Module):
    def __init__(self, global_in_dim: int, hidden_dim: int):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        node_agg = scatter_sum(x, batch, dim=0)
        out = torch.cat([u, node_agg], dim=1)
        return self.global_mlp(out)


class SDGNN(nn.Module):
    """Matches the SDGNN definition used in the notebook checkpoint."""

    def __init__(self, node_in_dim: int, edge_in_dim: int, global_in_dim: int, hidden_dim: int):
        super().__init__()
        self.meta_layer = MetaLayer(
            edge_model=EdgeModel(node_in_dim, edge_in_dim, hidden_dim),
            node_model=NodeModel(node_in_dim, hidden_dim),
            global_model=GlobalModel(global_in_dim, hidden_dim),
        )
        self.final_mlp = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        x, edge_attr, u = self.meta_layer(x, edge_index, edge_attr, u, batch)
        out = self.final_mlp(u)
        return out.squeeze(-1)
