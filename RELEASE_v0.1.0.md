## v0.1.0 — GamiBench Initial Public Release

First public release of GamiBench, a benchmark for evaluating spatial reasoning and 2D-to-3D planning in MLLMs using origami folding tasks.

Paper: https://arxiv.org/abs/2512.22207

### Highlights

- End-to-end script-based evaluation pipeline (not notebook-only)
- Three-task evaluation protocol:
  - Standard multiple-choice fold prediction
  - Alternative-viewpoint consistency
  - Impossible-fold detection
- Deterministic task generation with random seed control
- Checkpointing + resume support for long model runs
- Single-model and multi-model suite runners
- Open/closed model grouping support
- OpenRouter + closed-source provider integrations in unified framework
- Data layout simplified to `data/GamiBench`

### Included in this release

- Core benchmark/evaluator implementation for GamiBench
- Config-driven experiment execution
- Suite runner for evaluating one or many models
- Hugging Face publish/download helper scripts
- Updated README and docs for setup and usage

### Dataset

Use the Hugging Face dataset workflow documented in README, or your published dataset link.

### Quickstart

```bash
# single model
python run.py configs/experiments/gamibench_single.yaml

# multi-model suite
python scripts/run_gamibench_suite.py --config configs/experiments/gamibench_suite.yaml --group all
```

### Citation

```bibtex
@misc{spencer2025gamibenchevaluatingspatialreasoning,
      title={GamiBench: Evaluating Spatial Reasoning and 2D-to-3D Planning Capabilities of MLLMs with Origami Folding Tasks},
      author={Ryan Spencer and Roey Yaari and Ritvik Vemavarapu and Joyce Yang and Steven Ngo and Utkarsh Sharma},
      year={2025},
      eprint={2512.22207},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.22207},
}
```

### Notes

- API keys are expected via environment variables.
- This is an initial public release (`0.x`); interfaces may evolve in future versions.
