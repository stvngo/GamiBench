---
pretty_name: GamiBench
license: mit
task_categories:
- visual-question-answering
language:
- en
---

# GamiBench Dataset

GamiBench is a benchmark for evaluating spatial reasoning and 2D-to-3D planning in multimodal language models using origami folding tasks.

## Contents

- `data/GamiBench/`: image folders for each origami example.
- `configs/experiments/gamibench_single.yaml`: single-model evaluation config.
- `configs/experiments/gamibench_suite.yaml`: multi-model suite config.

## Local Usage

Use the pipeline in the main repository to run evaluations after downloading this dataset snapshot.

## Citation

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

