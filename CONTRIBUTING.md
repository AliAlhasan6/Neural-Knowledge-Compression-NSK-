# Contributing to NSK

Thank you for your interest in the NSK project. This document explains how to work on the codebase and the conventions used throughout.

---

## Development Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -r requirements.txt
pip install pytest black isort  # dev tools
```

## Running Tests

```bash
pytest tests/ -v
```

All 72 tests must pass before any PR can be merged.

## Code Style

- Formatter: `black` (line length 100)
- Import order: `isort`
- Type hints on all public functions
- Docstrings on all classes and non-trivial functions

Run before committing:
```bash
black src/ tests/
isort src/ tests/
```

## How the Stages Fit Together

If you change the output interface of any stage, you must update every downstream stage that consumes it:

```
compressor.py  →  model.py / train.py  →  merger.py / train_merger.py  →  joint_train.py  →  validate.py
```

The `MacroKnowledgeState` dataclass in `merger.py` is the shared contract between stages 3, 4, and 5. Any change to its fields must be reflected in all three.

## Adding a New Test

Tests live in `tests/`. Each file covers one stage:

| File | Stage |
|---|---|
| `test_compressor.py` | Stage 1 |
| `test_embedder.py` | Stage 2 |
| `test_merger.py` | Stage 3 |
| `test_joint.py` | Stage 4 |
| `test_multiagent.py` | Stage 5 |

Use minimal synthetic graphs (not FB15k-237) in tests so they run fast without dataset download:

```python
from torch_geometric.data import Data
import torch

def make_small_graph(num_nodes=6, num_edges=8, num_relations=5):
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type  = torch.randint(0, num_relations, (num_edges,))
    x          = torch.rand(num_nodes, 2)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)
```

## Priority Contributions

These are the highest-value areas for contribution:

1. **Learned GAT compressor** — Replace the heuristic four-signal scorer with a differentiable GAT-based learned compressor. The interface must remain `GraphCompressor.compress(graph: Data) → Data`.

2. **ROS2 integration** — Connect the pipeline to the scaffolded ROS2 workspace. A ROS2 node that wraps `SwarmAgent` and communicates via standard ROS2 topics would be ideal.

3. **Cross-attention merger (Option A)** — Implement the cross-attention fusion architecture as an alternative to the current gated update (Option C). Both should be selectable via `configs/base.yaml`.

4. **Synthetic swarm dataset** — Generate a domain-specific knowledge graph dataset (e.g., simulated sensor observations in a Gazebo environment) to complement FB15k-237 evaluation.

## Questions?

Open an issue with the `question` label, or contact the maintainer at Aliyossefalhasan@gmail.com.
