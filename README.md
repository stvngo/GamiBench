# GamiBench Research Pipeline

End-to-end evaluation pipeline for benchmark experiments, following ML Research Engineering best practices.

## 🏗️ Structure

```
.
├── configs/              # Configuration files (YAML/JSON)
│   ├── base.yaml        # Base configuration template
│   ├── experiments/     # Experiment-specific configs
│   ├── models/          # Model configurations
│   └── datasets/        # Dataset configurations
├── data/                # Datasets, preprocessing
│   ├── raw/            # Raw datasets
│   └── processed/      # Processed datasets
├── models/              # Model definitions, wrappers
│   ├── base.py         # BaseModel interface
│   └── model_factory.py # Model factory
├── evaluators/          # Evaluation logic
│   └── base.py         # BaseEvaluator interface
├── baselines/           # Baseline implementations
├── experiments/         # Experiment scripts
├── utils/               # Shared utilities
│   ├── config_loader.py
│   ├── logger.py
│   ├── seeding.py
│   ├── data_loader.py
│   └── result_saver.py
├── outputs/             # Results, logs, checkpoints
│   ├── results/
│   ├── logs/
│   └── checkpoints/
├── scripts/             # One-off scripts, analysis
├── pipeline.py          # Main pipeline orchestration
└── run.py              # Main entry point
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Option 1: Config-Driven (Recommended)

```bash
# Run with a config file
python run.py configs/experiments/example.yaml

# With overrides
python run.py configs/experiments/example.yaml \
    --override model.name=gpt-4 \
    --override evaluator.batch_size=10
```

#### Option 2: CLI-Driven

```bash
python run.py \
    --benchmark ... \
    --model gpt-4 \
    --model-config configs/models/openai.yaml
```

## 📝 Configuration

Configuration files use YAML format and support:
- Hierarchical configs (base + experiment-specific)
- Environment variables (`${VAR_NAME}`)
- Command-line overrides

### Example Config

```yaml
experiment_name: "my_experiment"
seed: 42

model:
  type: "openai"
  name: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.7

dataset:
  path: "data/my_dataset.json"
  format: "json"

evaluator:
  type: "my_benchmark"
  batch_size: 10

output_dir: "outputs/results"
```

## 🔧 Extending the Pipeline

### Adding a New Model

1. Create model class inheriting from `BaseModel`:

```python
# models/my_model.py
from .base import BaseModel

class MyModel(BaseModel):
    def generate(self, prompt, **kwargs):
        # Implementation
        pass
    
    def score(self, prompt, completion, **kwargs):
        # Implementation
        pass
```

2. Register in factory:

```python
# models/__init__.py
from .my_model import MyModel
ModelFactory.register("my_model", MyModel)
```

### Adding a New Evaluator

1. Create evaluator class inheriting from `BaseEvaluator`:

```python
# evaluators/my_benchmark.py
from .base import BaseEvaluator

class MyBenchmarkEvaluator(BaseEvaluator):
    def evaluate(self):
        results = self.evaluate_batch(self.data)
        metrics = self.compute_metrics(results)
        return {
            'results': results,
            'metrics': metrics
        }
    
    def evaluate_single(self, example):
        # Implementation
        pass
```

## 📊 Results

Results are saved in `outputs/results/` with:
- `results.json`: Full evaluation results
- `metrics.json`: Aggregated metrics
- `config.yaml`: Frozen configuration
- `metadata.json`: Experiment metadata

## 🧪 Best Practices

### Reproducibility
- All configs are saved with results
- Random seeds are set automatically
- Version control configs and code

### Error Handling
- Graceful degradation (skips bad examples)
- Comprehensive error logging
- Resume capability (checkpointing)

### Scalability
- Batch processing support
- Progress tracking
- Parallel processing where possible

## 📚 Documentation

- See `configs/base.yaml` for configuration options
- See `models/base.py` for model interface
- See `evaluators/base.py` for evaluator interface

## 🤝 Contributing

1. Follow the modular structure
2. Add docstrings to all functions
3. Write tests for new components
4. Update documentation

## 📄 License

[Your License Here]
