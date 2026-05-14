# NSK — Neuro-Symbolic Knowledge Graph Compression and Fusion

> **Distributed knowledge management for swarm robotics** — compressing, embedding, and fusing symbolic knowledge graphs across bandwidth-constrained multi-agent systems.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG 2.4](https://img.shields.io/badge/PyG-2.4-brightgreen)](https://pyg.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-blue?logo=mlflow)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Problem Does NSK Solve?

In a robot swarm, every agent sees only a local slice of the world. To act intelligently together, agents must **share what they know** — but real-world communication channels are narrow, bandwidth is scarce, and flooding the network with full knowledge graphs is infeasible.

NSK answers: *how do you compress a knowledge graph down to its most essential structure, encode it into a compact vector, and fuse those two representations so that a receiving agent learns as much as possible from what was sent?*

---

## Pipeline Architecture

```
Input KG (FB15k-237 ego-graph)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 1 · GraphCompressor  C_g: G → G̃                          │
│                                                                   │
│  Four-signal importance scoring:                                  │
│    PageRank (35%) + Semantic (25%) + Surprise (25%) + Recency (15%)│
│  + Bridge-edge preservation  + Non-expanding semantic closure     │
│  → retains ~42% of nodes/edges                                    │
└───────────────────────────────────────────────────────────────────┘
        │  G̃ (compressed symbolic graph)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 2 · KGEmbedder  ψ: G̃ → z ∈ ℝ³²                          │
│                                                                   │
│  GATv2Conv (2-layer, 2-head) graph autoencoder                    │
│  Relation-typed edge embeddings · Reconstruction loss             │
│  + Relation-type classification auxiliary loss                    │
└───────────────────────────────────────────────────────────────────┘
        │  z ∈ ℝ³²
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 3 · KnowledgeMerger  (G̃, z) → z*                         │
│                                                                   │
│  Graph-conditioned gated fusion (Option C):                       │
│    g_summary = GNN(G̃)                                            │
│    gate  g   = σ(W[z ‖ g_summary])                               │
│    z*        = g ⊙ z + (1−g) ⊙ g_summary                        │
│  Gate regularisation → 0.5 · Reconstruction anchor loss          │
└───────────────────────────────────────────────────────────────────┘
        │  MacroKnowledgeState(G̃, z*)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 4 · Joint End-to-End Fine-tuning                          │
│  Gradient flows: Merger loss → Embedder → (frozen Compressor)    │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 5 · Multi-Agent Swarm Simulation                          │
│  10 agents · 20 rounds · random + ring topologies                 │
│  Convergence: cosine similarity of z* increases monotonically     │
└───────────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Compression retention ratio | ~42% of original graph |
| Embedding dimension | 32 |
| Embedding variance before joint training | 0.0074 |
| Embedding variance after joint training | **0.0111** (+49% discriminability) |
| Communication topologies validated | Random, Ring, Fully-connected |
| Agents / rounds in swarm simulation | 10 agents / 20 rounds |
| Unit tests | **72 passing** |
| Training hardware | CPU (GTX 1050 4 GB, backward pass exceeds VRAM) |
| Total training time | ~30 h on i5-7300HQ |

### Why the 49% Discriminability Gain Matters

Before joint fine-tuning the merger, embeddings clustered tightly (variance = 0.0074) — the embedder had no signal from the symbolic structure. After end-to-end training, `z*` carries information from both the graph topology *and* the neural encoder, spreading representations apart (variance = 0.0111). A model that distinguishes graphs better is a model that shares more useful information.

---

## Stage-by-Stage Design Highlights

### Stage 1 — Graph Compressor (`src/stage1_compressor/compressor.py`)

The compressor is a **heuristic multi-signal scorer** — deliberately non-differentiable at this stage (a learned GAT compressor is the planned Phase 5 upgrade).

Each node receives a composite importance score:

```
importance(v) = 0.35 · PageRank(v)
              + 0.25 · SemanticCentrality(v)
              + 0.25 · InformationSurprise(v)
              + 0.15 · Recency(v)
```

Two structural safety mechanisms run after scoring:
- **Bridge-edge preservation** — Tarjan's algorithm detects edges whose removal disconnects the graph; these are kept unconditionally regardless of node scores.
- **Semantic closure (non-expanding)** — after node selection, any "dangling" class assertion is repaired by promoting the nearest instance node rather than adding new nodes.

### Stage 2 — GATv2 Embedder (`src/stage2_embedder/model.py`)

- **GATv2Conv** (Brody et al. 2022) — fixes the static attention problem of GAT v1; attention is recomputed per edge during message passing.
- Relation types are embedded as edge features, not just edge weights — the model learns that `/people/person/nationality` and `/film/film/genre` carry fundamentally different structural roles.
- Graph autoencoder loss: reconstruct both **edge existence** (binary) and **relation type** (multi-class) from the latent `z`.
- Optional **virtual node pooling** — one virtual node per graph aggregates global context before readout.

### Stage 3 — Gated Knowledge Merger (`src/stage3_merger/merger.py`)

The key design question was: *given a compressed graph and an embedding of that graph, how much should the final representation trust each source?*

The answer is a **learned, per-dimension gate**:

```python
# z:          [B, d]  from Stage 2 embedder
# g_summary:  [B, d]  from re-encoding G̃ with a lightweight GNN
gate    = torch.sigmoid(self.gate_net(torch.cat([z, g_summary], dim=-1)))
z_star  = gate * z + (1 - gate) * g_summary
```

Two regularisation terms prevent degenerate solutions:
1. **Gate centre loss** — `MSE(gate, 0.5)` — prevents the gate from collapsing to all-0 (ignore embedder) or all-1 (ignore graph).
2. **Reconstruction anchor loss** — the embedder's autoencoder loss continues to fire during joint training, preventing catastrophic forgetting of the encoding representation.

### Stage 5 — Multi-Agent Swarm Simulation (`src/multiagent/validate.py`)

The simulation reveals a non-obvious result: **ring topology converges better than fully-connected broadcasting**.

Under fully-connected topology, all agents simultaneously broadcast to all others. The merger receives a high-fanout average that saturates the gate, causing `z*` to collapse toward the global mean — agents lose their individual knowledge. The ring protocol (each agent shares with exactly one neighbour per round) preserves local diversity while still achieving convergence, with mean pairwise cosine similarity increasing monotonically over 20 rounds.

---

## Repository Structure

```
nsk/
├── configs/
│   └── base.yaml               # All hyperparameters in one place
├── src/
│   ├── stage1_compressor/
│   │   └── compressor.py       # Four-signal scorer + bridge detection
│   ├── stage2_embedder/
│   │   ├── model.py            # GATv2 autoencoder
│   │   └── train.py            # Training loop + evaluation
│   ├── stage3_merger/
│   │   ├── merger.py           # Gated fusion + MacroKnowledgeState
│   │   └── train_merger.py     # Merger training + ablation study
│   ├── joint_training/
│   │   └── joint_train.py      # End-to-end fine-tuning
│   ├── multiagent/
│   │   └── validate.py         # Swarm simulation (10 agents, 20 rounds)
│   └── utils/
│       └── data_loader.py      # FB15k-237 → ego-graph pipeline
├── tests/
│   ├── test_compressor.py
│   ├── test_embedder.py
│   ├── test_joint.py
│   ├── test_merger.py
│   └── test_multiagent.py
├── experiments/
│   └── mlruns/                 # MLflow tracking (11 runs logged)
├── docs/
│   ├── ARCHITECTURE.md         # Deep technical reference
│   └── RESULTS.md              # Full experimental results
└── paper/
    └── NSK_IEEE_Paper.pdf      # IEEE-style technical paper
```

---

## Quick Start

### 1. Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# PyTorch with CUDA 11.8 (GTX 1050 compatible)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Remaining dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python -c "
from torch_geometric.datasets import FB15k_237
FB15k_237(root='data/raw')
print('Dataset ready.')
"
```

### 3. Run the Full Pipeline

```bash
# Stage 2: Train the embedder (anchors the whole pipeline)
python -m src.stage2_embedder.train

# Stage 3: Train the merger
python -m src.stage3_merger.train_merger

# Stage 4: Joint end-to-end fine-tuning
python -m src.joint_training.joint_train

# Stage 5: Multi-agent swarm validation
python -m src.multiagent.validate
```

### 4. Run Tests

```bash
pytest tests/ -v
# Expected: 72 passed
```

### 5. View MLflow Experiments

```bash
mlflow ui --backend-store-uri experiments/mlruns
# Open http://127.0.0.1:5000
```

---

## Configuration

All hyperparameters live in `configs/base.yaml`:

```yaml
compressor:
  retention_ratio: 0.40        # Keep 40% of nodes
  w_structural: 0.35           # PageRank weight
  w_semantic: 0.25             # Semantic centrality weight
  w_surprise: 0.25             # Information-theoretic surprise weight
  w_recency: 0.15              # Recency weight

embedder:
  hidden_dim: 32
  output_dim: 32               # z ∈ ℝ³²
  num_layers: 2
  heads: 2

merger:
  architecture: C              # Gated vector update
  use_gating: true

hardware:
  device: cpu                  # Backward pass exceeds GTX 1050 VRAM
```

---

## Dataset: FB15k-237

| Property | Value |
|---|---|
| Total triples | 310,116 |
| Entities | 14,541 |
| Relation types | 237 |
| Ego-graphs sampled | 5,000 (k=2 hop) |
| Valid graphs (≥4 nodes, ≥3 edges) | 4,530 |
| Train / Val / Test split | 3,171 / 679 / 680 |
| Node feature dimensionality | 2 (norm. ID, norm. degree) |

FB15k-237 was chosen over NELL, ConceptNet, and OGB for its balance of benchmarkability, typed relational diversity, and manageable size for iterative development on constrained hardware. Ego-graphs (k=2 hop subgraphs centred on a sampled entity) mirror the per-agent local knowledge snapshot in a real swarm setting.

---

## Limitations & Planned Extensions

| Item | Status | Notes |
|---|---|---|
| Learned GAT compressor | 🔲 Planned (Phase 5) | Replace heuristic scorer with end-to-end learned node selection |
| ROS2 integration | 🔲 Planned | Workspace scaffolded; not yet connected to pipeline |
| Real swarm experiments | 🔲 Future work | Requires physical or Gazebo hardware |
| Cross-attention merger (Option A) | 🔲 Future work | Update both z and G̃ simultaneously |
| Quantisation-aware training | 🔲 Future work | Reduce transmission bandwidth further |
| Dynamic topologies | 🔲 Future work | Agents joining/leaving mid-run |
| Domain-specific dataset | 🔲 Future work | Synthetic swarm-domain KG to replace FB15k-237 |

---

## Citation

If you use NSK in your research, please cite:

```bibtex
@article{alhasan2026nsk,
  title     = {Neuro-Symbolic Knowledge Graph Compression and Fusion
               for Distributed Swarm Robotics},
  author    = {Alhasan, Ali and Viksnin, I.I.},
  journal   = {IEEE [venue TBD]},
  year      = {2026},
  institution = {Saint Petersburg Electrotechnical University (LETI),
                 Swarm Intelligence \& Knowledge Systems Laboratory}
}
```

---

## License

MIT © 2026 Alhasan Ali, LETI Swarm Intelligence & Knowledge Systems Laboratory
