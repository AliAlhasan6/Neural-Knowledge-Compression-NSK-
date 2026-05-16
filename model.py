"""
src/stage2_embedder/model.py

Stage 2: Knowledge Graph Embedder  ψ: G → z ∈ R^d

Architecture:
  1. Node initialisation  — feature projection (input_dim → hidden_dim)
  2. Message passing      — L layers of GATv2Conv (attention-weighted)
  3. Readout / pooling    — mean | sum | virtual_node → graph vector z

The model is relation-type aware: edge types are embedded and
concatenated to messages during aggregation, so the GNN distinguishes
between  isA  and  observedAt  rather than treating all edges equally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops


class RelationEmbedding(nn.Module):
    """
    Embeds discrete relation types into a continuous vector.
    Used to make message passing relation-aware.
    """
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_relations + 1, dim, padding_idx=0)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.emb(edge_type + 1)  # shift for padding_idx


class KGEmbedder(nn.Module):
    """
    Full Stage 2 embedder.

    Args:
        input_dim:      node feature dimension (2 for our FB15k setup)
        hidden_dim:     internal representation width
        output_dim:     graph embedding dimension d
        num_layers:     GNN depth L
        num_relations:  number of distinct relation types in the dataset
        heads:          GAT attention heads
        dropout:        dropout rate
        pooling:        'mean' | 'sum' | 'virtual_node'
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        num_relations: int = 237,
        heads: int = 4,
        dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        # --- Node feature projection ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- Relation type embeddings ---
        # Concatenated to source node before GAT aggregation
        self.rel_emb = RelationEmbedding(num_relations, hidden_dim)

        # --- GATv2 message passing layers ---
        # GATv2 is strictly more expressive than GATv1 (can distinguish
        # any two distinct neighbourhoods; GATv1 cannot in some cases)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels  = hidden_dim * (heads if i > 0 else 1)
            out_channels = hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=True,   # concatenate heads → hidden_dim * heads
                    add_self_loops=True,
                    edge_dim=hidden_dim  # relation embedding injected here
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # --- Virtual node (optional pooling) ---
        # vn_emb must match hidden_dim (pre-layer size), NOT hidden_dim*heads.
        # _inject_virtual_node is called before message passing, so h is still
        # [N, hidden_dim]. The VN gets expanded to hidden_dim*heads by the
        # first GATv2 layer alongside all other nodes.
        if pooling == 'virtual_node':
            self.vn_emb  = nn.Embedding(1, hidden_dim)
            self.vn_proj = nn.Linear(hidden_dim * heads, hidden_dim * heads)
            nn.init.zeros_(self.vn_emb.weight)

        # --- Output projection: flatten multi-head → output_dim ---
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_type, batch):
        """
        Args:
            x:          [N, input_dim]   node features
            edge_index: [2, E]           edge connectivity
            edge_type:  [E]              integer relation type per edge
            batch:      [N]              batch assignment vector

        Returns:
            z:  [B, output_dim]   graph-level embeddings
            h:  [N, hidden_dim*heads]  final node representations
        """
        # Project node features
        h = F.relu(self.input_proj(x))    # [N, hidden_dim]

        # Relation embeddings for all edges
        edge_attr = self.rel_emb(edge_type)  # [E, hidden_dim]

        # Virtual node: add one VN per graph in batch
        if self.pooling == 'virtual_node':
            h, edge_index, edge_attr, batch, vn_h = \
                self._inject_virtual_node(h, edge_index, edge_attr, batch)

        # --- Message passing ---
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_attr=edge_attr)  # [N, H*heads]
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection from layer 1 onward (shapes match)
            if i > 0:
                h_new = h_new + h
            h = h_new

        # --- Readout ---
        if self.pooling == 'virtual_node':
            z_pool = self._read_virtual_node(h, batch, vn_h)
        elif self.pooling == 'sum':
            z_pool = global_add_pool(h, batch)     # [B, H*heads]
        else:  # mean (default)
            z_pool = global_mean_pool(h, batch)    # [B, H*heads]

        # Project to output_dim
        z = self.output_proj(z_pool)               # [B, output_dim]
        return z, h

    # ------------------------------------------------------------------
    # Virtual node helpers
    # ------------------------------------------------------------------

    def _inject_virtual_node(self, h, edge_index, edge_attr, batch):
        """
        Add one virtual node per graph. Connect it to all real nodes.
        The VN aggregates global information each layer.
        """
        device = h.device
        num_nodes = h.size(0)
        batch_size = batch.max().item() + 1

        # Virtual node features (one per graph)
        vn_h = self.vn_emb(
            torch.zeros(batch_size, dtype=torch.long, device=device)
        )  # [B, H*heads] — start as zeros

        # Assign VN indices: offset beyond real nodes
        vn_idx = torch.arange(batch_size, device=device) + num_nodes

        # Build VN ↔ real-node edges (bidirectional)
        real_nodes = torch.arange(num_nodes, device=device)
        vn_per_node = vn_idx[batch]

        vn_edges_fwd = torch.stack([vn_per_node, real_nodes], dim=0)
        vn_edges_bwd = torch.stack([real_nodes, vn_per_node], dim=0)
        new_edges = torch.cat([vn_edges_fwd, vn_edges_bwd], dim=1)

        # Dummy edge attr for VN edges (zeros)
        vn_attr = torch.zeros(new_edges.size(1), edge_attr.size(1),
                               device=device)

        # Concatenate
        h_aug        = torch.cat([h, vn_h], dim=0)
        edge_index_aug = torch.cat([edge_index, new_edges], dim=1)
        edge_attr_aug  = torch.cat([edge_attr, vn_attr], dim=0)
        batch_aug    = torch.cat([batch,
                                   torch.arange(batch_size, device=device)])

        return h_aug, edge_index_aug, edge_attr_aug, batch_aug, vn_h

    def _read_virtual_node(self, h_aug, batch_aug, vn_h):
        """
        Extract the virtual node embedding for each graph as the readout.
        VNs are the last batch_size entries in h_aug.
        """
        batch_size = vn_h.size(0)
        return h_aug[-batch_size:]  # [B, H*heads]


class GraphAutoencoder(nn.Module):
    """
    Wraps the embedder with a decoder for reconstruction-based training.

    The decoder reconstructs edge existence (and optionally relation type)
    from pairs of node embeddings — standard graph autoencoder objective.

    Loss = Binary cross-entropy on edge reconstruction
         + Cross-entropy on relation type prediction (optional)
    """

    def __init__(self, embedder: KGEmbedder, num_relations: int = 237):
        super().__init__()
        self.embedder = embedder
        d = embedder.hidden_dim * embedder.convs[0].heads

        # Edge existence decoder: dot product between node pairs
        # (no parameters needed — inner product is expressive enough)

        # Relation type decoder: MLP on concatenated node pairs
        self.rel_decoder = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Linear(d, num_relations)
        )

    def forward(self, x, edge_index, edge_type, batch):
        z, h = self.embedder(x, edge_index, edge_type, batch)
        return z, h

    def reconstruction_loss(self, h, edge_index, edge_type,
                             num_nodes: int, neg_ratio: int = 1):
        """
        Computes edge reconstruction loss.

        Positive edges: existing edges in edge_index
        Negative edges: randomly sampled non-edges

        Args:
            h:          [N, d]  node representations
            edge_index: [2, E]  positive edges
            edge_type:  [E]     relation labels for positive edges
            num_nodes:  N
            neg_ratio:  negative samples per positive edge

        Returns:
            loss_exist: edge existence BCE loss
            loss_rel:   relation type CE loss
        """
        device = h.device
        E = edge_index.size(1)

        # --- Positive edge scores ---
        src_pos = h[edge_index[0]]   # [E, d]
        dst_pos = h[edge_index[1]]   # [E, d]
        scores_pos = (src_pos * dst_pos).sum(dim=-1)  # [E]

        # --- Negative edge sampling ---
        neg_src = torch.randint(0, num_nodes, (E * neg_ratio,), device=device)
        neg_dst = torch.randint(0, num_nodes, (E * neg_ratio,), device=device)
        src_neg = h[neg_src]
        dst_neg = h[neg_dst]
        scores_neg = (src_neg * dst_neg).sum(dim=-1)  # [E*r]

        # BCE loss
        labels = torch.cat([
            torch.ones(E, device=device),
            torch.zeros(E * neg_ratio, device=device)
        ])
        scores = torch.cat([scores_pos, scores_neg])
        loss_exist = F.binary_cross_entropy_with_logits(scores, labels)

        # --- Relation type prediction (positive edges only) ---
        pair_feat = torch.cat([src_pos, dst_pos], dim=-1)  # [E, 2d]
        rel_logits = self.rel_decoder(pair_feat)            # [E, num_rel]
        loss_rel = F.cross_entropy(rel_logits, edge_type)

        return loss_exist, loss_rel


def build_model(config: dict, num_relations: int = 237) -> GraphAutoencoder:
    """
    Instantiates the full autoencoder from config.
    """
    ec = config['embedder']
    embedder = KGEmbedder(
        input_dim=2,
        hidden_dim=ec['hidden_dim'],
        output_dim=ec['output_dim'],
        num_layers=ec['num_layers'],
        num_relations=num_relations,
        heads=ec['heads'],
        dropout=ec['dropout'],
        pooling=ec['pooling']
    )
    model = GraphAutoencoder(embedder, num_relations=num_relations)
    return model
