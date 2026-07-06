# AmbiDrop — Documentation Index

_Last updated: 2026-07-06_

Overview of all documentation files in this repository and what each one covers.

---

## Published Documentation

These files are committed to the repository and intended for readers of the codebase.

| File | Audience | Contents |
|------|----------|----------|
| [README.md](README.md) | Everyone | Project overview, method description, pipeline figures, quick start, wrapper script flags, code→paper mapping, ablation descriptions |
| [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md) | Developers | Neural network architectures, data pipeline (types A/B/C), preprocessing functions, checkpoint table, ASM formula and API. **Keep this up to date when the code changes.** |
| [USAGE.md](USAGE.md) | Users / developers | Complete CLI reference for every script: data generators, wrapper scripts, direct train/eval scripts, ablation scripts, ASM Python API, checkpoint registry |
| [DOCS.md](DOCS.md) | Everyone | This file — index of all documentation |

---

## Update Policy

| File | When to update |
|------|---------------|
| `README.md` | When the project description, method, or public interface changes |
| `CODEBASE_OVERVIEW.md` | **After every working session that modifies the codebase** — new files, renamed functions, new checkpoints, changed APIs |
| `USAGE.md` | When CLI flags, script behaviour, or the Python API changes |
| `DOCS.md` | When a new documentation file is added or removed |

Each file carries a `_Last updated: YYYY-MM-DD_` line at the top. Update it when you edit the file.

---

## Quick Navigation

**I want to…**

- Understand what AmbiDrop does → [README.md](README.md)
- Run the full pipeline → [README.md § Quick Start](README.md#quick-start)
- See all CLI flags for a script → [USAGE.md](USAGE.md)
- Understand the neural network architecture → [CODEBASE_OVERVIEW.md § 1. Neural Networks](CODEBASE_OVERVIEW.md#1-neural-networks)
- Understand the data pipeline → [CODEBASE_OVERVIEW.md § 2. Data Pipeline](CODEBASE_OVERVIEW.md#2-data-pipeline)
- Use the ASM Python API → [USAGE.md § 5. ASM](USAGE.md#5-asm-ambisonics-signal-matching)
- Load / save checkpoints → [USAGE.md § 11. Checkpoint Registry](USAGE.md#11-checkpoint-registry)
- Run ablation experiments → [USAGE.md § 10. Ablation Scripts](USAGE.md#10-ablation-scripts)
- Reproduce paper results → [README.md § Code → Paper Mapping](README.md#code--paper-mapping)
