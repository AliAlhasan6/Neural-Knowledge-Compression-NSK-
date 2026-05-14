# NSK Architecture — Deep Technical Reference

This document explains every design decision in the NSK pipeline in depth. The README gives you the "what"; this document gives you the "why".

---

## Table of Contents

1. [Mathematical Formulation](#1-mathematical-formulation)
2. [Stage 1 — Graph Compressor](#2-stage-1--graph-compressor)
3. [Stage 2 — GATv2 Embedder](#3-stage-2--gatv2-embedder)
4. [Stage 3 — Gated Knowledge Merger](#4-stage-3--gated-knowledge-merger)
5. [Joint End-to-End Training](#5-joint-end-to-end-training)
6. [Multi-Agent Swarm Protocol](#6-multi-agent-swarm-protocol)
7. [Training Stability Mechanisms](#7-training-stability-mechanisms)
8. [Hardware Constraints and Engineering Decisions](#8-hardware-constraints-and-engineering-decisions)

---

## 1. Mathematical Formulation

### Agent State

Each agent `i` maintains a **MacroKnowledgeState**:

```
MKS_i = (G̃_i, z*_i)
```

where:
- `G̃_i ⊂ G_i` — compressed knowledge graph (symbolic)
- `z*_i ∈ ℝ^d` — fused macro-knowledge embedding (neural)

### Per-Round Update

At each communication round `t`:

```
G̃_i(t)  = C_g(G_i(t))                        # Stage 1: compress
z_i(t)   = ψ(G̃_i(t))                          # Stage 2: embed
z*_i(t)  = Φ(z_i(t), G̃_i(t))                 # Stage 3: fuse

# After receiving z̃_j from neighbour j:
z*_i(t+1) = 0.7 · z*_i(t) + 0.3 · z̃_j(t)    # Weighted average merge
```

### Swarm-Level Macro-Knowledge

The aggregate swarm state (an analytical quantity, not computed by any individual agent):

```
G_swarm(t)  = ∪_{i=1}^{N} G̃_i(t)
z_swarm(t)  = (1/N) Σ_i z*_i(t)
```

---

## 2. Stage 1 — Graph Compressor

**File:** `src/stage1_compressor/compressor.py`

### Why Heuristic, Not Learned?

A differentiable node-selection mechanism (e.g., a soft top-k over node scores) was considered but deferred. The reasons:

1. Straight-through estimators for hard node selection introduce gradient noise that destabilised early joint training.
2. The heuristic compressor provides a stable, interpretable baseline against which a future learned compressor can be ablated.
3. Bridge-edge preservation is a *hard structural constraint* — it must hold regardless of gradient signal.

The planned Phase 5 replacement is a GAT-based learned compressor that outputs continuous importance weights and uses a differentiable top-k (or Gumbel-softmax relaxation).

### Four-Signal Importance Scoring

```python
importance(v) = w_s · PageRank(v)      # structural centrality
              + w_e · Semantic(v)      # ontological role
              + w_p · Surprise(v)      # information gain
              + w_r · Recency(v)       # temporal freshness
```

**PageRank (w=0.35):** Nodes connected to many other important nodes score higher. Uses `alpha=0.85` (standard damping). Falls back to degree centrality if power iteration fails to converge (common on very small ego-graphs).

**Semantic Centrality (w=0.25):** Class-level nodes (e.g., type nodes in an OWL ontology) score 1.0; instance nodes score 0.5. In FB15k-237, explicit ontology labels are absent, so high-in-degree nodes are used as a proxy for class-level status.

**Information-Theoretic Surprise (w=0.25):** Rare relation types carry more information per Shannon entropy. A node incident to many rare-relation edges scores higher:

```
P(r) = count(r) / total_edges
surprise(v) = mean_{e incident to v} [ -log P(type(e)) ]
```

**Recency (w=0.15):** In FB15k-237 there are no real timestamps; recency is simulated via normalised node ID (a proxy for observation order). This signal is meaningful when the compressor is deployed on a real swarm with time-stamped sensor observations.

### Bridge-Edge Preservation

After node selection, the algorithm uses **Tarjan's bridge-finding algorithm** (`networkx.bridges`) to identify edges whose removal would disconnect the graph. These edges — and their endpoint nodes — are re-inserted unconditionally, overriding the node budget.

This prevents the compressed graph from fragmenting into disconnected components, which would break downstream path-based reasoning.

### Semantic Closure (Non-Expanding)

After pruning, some class assertions may be "dangling" — their object node was pruned but the subject node was retained. The closure repair algorithm:

1. Identifies dangling assertions.
2. Finds the nearest non-pruned node that can serve as a substitute class representative.
3. Redirects the assertion — it does **not** add new nodes to the graph.

"Non-expanding" is the key constraint: the closure step cannot increase the node count beyond the retention budget, only reroute edges.

---

## 3. Stage 2 — GATv2 Embedder

**File:** `src/stage2_embedder/model.py`

### Why GATv2 Over GAT?

Standard GAT (Veličković et al. 2018) computes attention as:

```
e(h_i, h_j) = a · LeakyReLU(W·h_i ‖ W·h_j)
```

The attention weight depends on the *concatenation order* of source and target features but not on the specific edge. This means that for a given node `i`, attention weights depend only on `h_i`, not on the combination `(h_i, h_j)` — GAT attention is **static** with respect to the key node.

GATv2 (Brody et al. 2022) fixes this:

```
e(h_i, h_j) = a · LeakyReLU(W · [h_i ‖ h_j])
```

Both source and target are projected *before* the nonlinearity, making attention a function of the specific (source, target) pair. This is strictly more expressive and empirically outperforms GAT on heterogeneous graphs like FB15k-237.

### Relation-Typed Edge Embeddings

Rather than encoding relation types as scalar edge weights, each relation type `r` is assigned a learned embedding vector:

```python
self.rel_emb = nn.Embedding(num_relations + 1, hidden_dim, padding_idx=0)
```

These embeddings are passed as `edge_attr` to `GATv2Conv`, which internally concatenates them to node features during attention computation. The model therefore learns that `/people/person/nationality` and `/film/film/genre` represent structurally different roles in the knowledge graph.

### Graph Autoencoder Loss

The embedder is trained to reconstruct the input graph from `z`:

```
L_recon = L_edge + λ · L_relation
```

- **L_edge:** Binary cross-entropy over all (i, j) pairs — does edge (i,j) exist?
- **L_relation:** Cross-entropy over existing edges — what relation type is (i,j)?

This dual objective ensures `z` captures both **connectivity structure** (which nodes are linked) and **relational semantics** (how they are linked).

### Virtual Node Pooling (Optional)

When `pooling='virtual_node'`, one additional node per graph is inserted that connects to all other nodes. After message passing, only this node's representation is used for graph-level readout. Virtual nodes help the pooling step aggregate long-range dependencies that might be missed by mean pooling on large graphs.

---

## 4. Stage 3 — Gated Knowledge Merger

**File:** `src/stage3_merger/merger.py`

### Fusion Architecture Choice

Three architectures were considered:

| Option | Description | Why Not Chosen |
|---|---|---|
| A | Cross-attention between graph nodes and z | Highest expressivity, but requires simultaneous update of both G̃ and z. Adds >3× parameters. Planned for Phase 5. |
| B | Concatenate z with mean pooling of G̃, project | Loses per-node structure; mean pooling discards topology. |
| **C** | Graph-conditioned gated update of z (chosen) | Balances expressivity with parameter efficiency. Gate adapts per-dimension. |

### Option C in Detail

```
g_summary = CompressedGraphEncoder(G̃)     # [B, d]
cat        = [z ‖ g_summary]               # [B, 2d]
gate g     = σ( W_gate · cat + b_gate )   # [B, d] ∈ (0,1)
z*         = g ⊙ z + (1−g) ⊙ g_summary   # [B, d]
```

The gate is **per-dimension**: the merger can choose to trust the neural embedding for some latent dimensions and the graph summary for others. For example, structural topology features (which tend to be well-captured by the GNN summary) can be given higher graph weight, while relational semantics (well-captured by the autoencoder embedding) can be given higher embedding weight.

### CompressedGraphEncoder

The re-encoder of `G̃` is architecturally identical to the Stage 2 embedder but smaller (`hidden_dim=32`, `num_layers=2`). It is kept separate so that:

1. It can be trained with a different learning rate from the embedder.
2. It can be frozen independently during ablation studies.
3. Its weights do not interfere with the Stage 2 checkpoint.

---

## 5. Joint End-to-End Training

**File:** `src/joint_training/joint_train.py`

### Gradient Flow

```
Merger loss → z* → gate network → g_summary / CompressedGraphEncoder → G̃
                               → z → Embedder (GATv2)
```

The compressor is **frozen** during joint training (non-differentiable heuristic). Gradient flows through the embedder and merger only.

### Loss Function

```
L_joint = L_merger + α · L_recon_anchor
```

- **L_merger:** The primary objective — measures whether `z*` faithfully represents the knowledge in `(G̃, z)`. Implemented as variance maximisation (push `z*` values to be distinguishable across graphs) plus a contrastive term.
- **L_recon_anchor (α=0.3):** The Stage 2 reconstruction loss, applied to the embedder weights even during joint training. Prevents catastrophic forgetting — without this anchor, the embedder drifts away from the autoencoder objective and `z` loses its geometric structure.

### Why the Reconstruction Anchor?

In preliminary runs without the anchor loss, joint training improved `var(z*)` in early epochs but caused the embedder to collapse (all graphs mapped to similar `z` vectors). The merger compensated by ignoring `z` and relying almost entirely on `g_summary`, which then had to carry all the information. The anchor loss keeps the embedder honest.

---

## 6. Multi-Agent Swarm Protocol

**File:** `src/multiagent/validate.py`

### Communication Topologies

Three topologies were tested:

**Random:** Each agent selects one random neighbour per round. Asymmetric — agent A may send to agent B without B sending back. Converges well because information diffuses stochastically across the swarm.

**Ring:** Agents are arranged in a fixed ring; each agent sends to its right neighbour only. Highly structured diffusion. Converges well with preserved local diversity — no agent's `z*` saturates because each receives information from exactly one source per round.

**Fully-connected:** Every agent broadcasts to all others simultaneously. **Fails to converge usefully.** See below.

### The Fully-Connected Saturation Failure

Under fully-connected broadcasting, each agent's `z*` is updated as:

```
z*_i(t+1) = 0.7 · z*_i(t) + 0.3 · mean_{j≠i}(z̃_j(t))
```

With N=10 agents, the mean term is an average of 9 embedding vectors. At convergence, all agents carry approximately the same global mean, which is also the initial global mean — no information has been *gained*. The gate in the merger saturates because the incoming signal is always the global average, which is consistent but uninformative.

**Practical implication:** Real swarm deployments should use ring or sparse random protocols rather than broadcasting. This is a concrete operational guideline that falls directly out of the simulation.

### Convergence Metric

```python
similarity_matrix[i,j] = cosine_similarity(z*_i, z*_j)
mean_pairwise_sim(t) = mean over all i≠j of similarity_matrix[i,j]
```

A monotonically increasing `mean_pairwise_sim` over rounds indicates that agents are building a shared macro-knowledge representation.

---

## 7. Training Stability Mechanisms

### Gate Centre Regularisation

```python
L_gate_reg = MSE(gate, torch.full_like(gate, 0.5))
```

Without this term, the gate can collapse in two ways:
- **All-1 gate:** The merger ignores `g_summary` entirely and just passes `z` through. The merger degenerates to an identity function.
- **All-0 gate:** The merger ignores `z` and outputs `g_summary` entirely. The Stage 2 embedder becomes irrelevant.

The centre regularisation keeps the gate near 0.5 at initialisation and provides a soft restoring force throughout training, while still allowing the gate to deviate toward either extreme when justified by the data.

### Layer Normalisation in GATv2

Each GATv2 layer is followed by `nn.LayerNorm`, applied to the concatenated multi-head output. This stabilises training on ego-graphs of highly variable size (4 to ~200 nodes in the FB15k-237 dataset). Without normalisation, very large graphs produce large activations that destabilise the gate network.

### Warmup Schedule

The embedder uses a 5-epoch linear warmup before the main learning rate schedule. On the first few batches, the randomly-initialised relation embeddings produce noisy gradients; the warmup prevents these from corrupting the attention weights before they have had a chance to learn meaningful relation distinctions.

---

## 8. Hardware Constraints and Engineering Decisions

The system was developed on:
- CPU: Intel i5-7300HQ
- GPU: NVIDIA GTX 1050 (4 GB VRAM)

### CPU-Only Training

The forward pass of the full pipeline fits within 4 GB VRAM. The backward pass does not — PyTorch retains all intermediate activations for gradient computation, roughly doubling peak memory.

Solution: `device: cpu` in `configs/base.yaml`. CPU training runs at ~4–6 hours per 100 epochs on the i5-7300HQ, which is acceptable for research-scale experiments.

This is documented explicitly rather than hidden, because it establishes an honest baseline: **the system works on consumer hardware**, which is an important property for swarm robotics research where edge-deployed agents will have similarly constrained hardware.

### Embedding Dimension Reduction

The design target was `d=128`. VRAM constraints forced a reduction to `d=32`. The 49% discriminability improvement after joint training holds at `d=32`; it is likely that `d=128` would show a larger absolute improvement, but this has not been validated.

### Batch Size

Batch size `B=8` ego-graphs. Larger batches would accelerate training but require more RAM for the graph adjacency structures. 8 was the maximum that kept RAM usage below 12 GB on the development machine.
