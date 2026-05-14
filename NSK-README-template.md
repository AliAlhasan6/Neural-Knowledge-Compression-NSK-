# NSK — Neuro-Symbolic Knowledge System

> A three-stage neuro-symbolic pipeline for compressing, sharing, and fusing knowledge graphs across distributed multi-agent systems.

<!-- Replace these badge placeholders once you have CI and a license -->
![Status](https://img.shields.io/badge/status-research-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-[CHOOSE]-lightgrey)

---

## Overview

NSK is a neuro-symbolic system that lets independent agents reason over a shared world through compressed knowledge-graph embeddings. The architecture has three stages:

1. **Graph Compressor** — reduces each agent's local knowledge graph to a compact set of structurally meaningful nodes and relations.
2. **GATv2 Embedder** — produces dense embeddings using a Graph Attention Network (v2).
3. **Gated Merger** — fuses incoming embeddings from peer agents under learned gating, with optional confidence weighting and conflict resolution.

The system targets two long-standing problems in multi-agent reasoning: *how to communicate knowledge cheaply* and *how to reconcile divergent observations* (subjectivity, conflict, trust).

This repository accompanies a first-author paper currently in revision for **IEEE SCM 2026** (full citation below once accepted).

---

## Architecture

```
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Knowledge   │ ──▶ │ Graph Compressor │ ──▶ │    GATv2     │
│     Graph     │     │  (per-agent)     │     │   Embedder   │
└───────────────┘     └──────────────────┘     └──────┬───────┘
                                                      │
                          peer embeddings ────▶ ┌─────▼──────┐
                                                │   Gated    │
                                                │   Merger   │
                                                └─────┬──────┘
                                                      │
                                              ┌───────▼────────┐
                                              │ Fused Knowledge│
                                              │  for agent i   │
                                              └────────────────┘
```

For a detailed treatment of subjectivity parameterisation, novelty detection, and trust management, see [`docs/theory.md`](docs/theory.md). *(to be added)*

---

## Results

Trained and validated on the **FB15k-237** benchmark:

| Metric                                  | Value     |
| --------------------------------------- | --------- |
| Mean pairwise graph similarity (fused)  | **0.72**  |
| Joint training — best checkpoint epoch  | **28**    |
| Unit and integration tests passing      | **72 / 72** |

*[Add: ablation table, comparison against single-agent baseline, per-relation breakdown.]*

---

## Quick start

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA (CPU works, slowly)
- PyTorch Geometric

### Installation

```bash
git clone https://github.com/AliAlhasan6/NSK.git
cd NSK
python -m venv nsk_env
source nsk_env/bin/activate
pip install -r requirements.txt
```

### Running

```bash
# Reproduce the joint-training run on FB15k-237
python scripts/train_joint.py --config configs/fb15k237.yaml

# Run inference with a saved checkpoint
python scripts/inference.py --checkpoint checkpoints/joint_best.pt \
                            --graph data/fb15k237/test.tsv
```

*[Add: full CLI reference, expected output shapes, GPU memory requirements.]*

---

## Repository layout

```
NSK/
├── nsk/                  # core library
│   ├── compressor.py     # Stage 1 — graph compression
│   ├── embedder.py       # Stage 2 — GATv2 embeddings
│   └── merger.py         # Stage 3 — gated fusion
├── scripts/              # training & inference entry points
├── configs/              # YAML configs per experiment
├── tests/                # unit and integration tests (72 passing)
├── docs/                 # theory, design notes
└── checkpoints/          # saved models (gitignored)
```

---

## Companion project

NSK is deployed across a multi-agent simulation in **[NSKsim](https://github.com/AliAlhasan6/NSKsim)** — a ROS 2 Jazzy / Gazebo Harmonic environment with 5–8 differential-drive robots sharing compressed knowledge over ZMQ inter-process sockets. *(coming soon)*

---

## Citation

If you use NSK in academic work, please cite:

```bibtex
@inproceedings{alhasan2026nsk,
  author    = {Alhasan, Ali and Viksnin, Ilya I.},
  title     = {[PAPER TITLE — fill in once final]},
  booktitle = {Proceedings of the IEEE SCM 2026},
  year      = {2026},
  note      = {In revision}
}
```

---

## Author

**Ali Alhasan** — PhD candidate, Saint Petersburg Electrotechnical University (LETI)
Supervisor: **Dr. Ilya I. Viksnin**

- ORCID: [0009-0007-1496-2736](https://orcid.org/0009-0007-1496-2736)
- ✉️ aliyossefalhasan@gmail.com

---

## License

[CHOOSE: MIT recommended for code; Apache-2.0 if you want explicit patent protection; CC-BY-4.0 if you want to share docs/data separately]

---

## Acknowledgments

This work was carried out at the AI / ML group, Saint Petersburg Electrotechnical University (LETI), as part of doctoral research toward a candidate-of-sciences degree in specialty 1.2.1 (Artificial Intelligence and Machine Learning).
