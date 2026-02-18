# Quick Start Guide

## 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API keys (if using OpenAI)
export OPENAI_API_KEY="your-key-here"
```

## 2. Prepare Your Data

Place your dataset in `data/raw/` or `data/processed/`:

```json
[
  {
    "id": "example_1",
    "prompt": "Your prompt here",
    "expected": "Expected output (optional)"
  },
  ...
]
```

## 3. Create Your Evaluator

Copy `evaluators/example_evaluator.py` and modify:

```python
# evaluators/your_benchmark.py
from evaluators.base import BaseEvaluator

class YourBenchmarkEvaluator(BaseEvaluator):
    def evaluate_single(self, example):
        prompt = example['prompt']
        response = self.model.generate(prompt)
        score = self._your_scoring_logic(example, response)
        return {
            'example_id': example['id'],
            'response': response,
            'score': score,
            'success': True
        }
```

## 4. Create Config

Create `configs/experiments/your_experiment.yaml`:

```yaml
base_config: "../base.yaml"

experiment_name: "your_experiment"

model:
  type: "openai"
  name: "gpt-4"
  api_key: "${OPENAI_API_KEY}"

dataset:
  path: "data/your_dataset.json"
  format: "json"

evaluator:
  type: "your_benchmark"  # Matches YourBenchmarkEvaluator
```

## 5. Run Experiment

```bash
python run.py configs/experiments/your_experiment.yaml
```

## 6. View Results

Results are saved in `outputs/results/{experiment_name}_{timestamp}/`:
- `results.json`: Full results
- `metrics.json`: Aggregated metrics
- `config.yaml`: Frozen config
- `metadata.json`: Experiment metadata

## Common Commands

```bash
# Run with overrides
python run.py configs/experiments/your_experiment.yaml \
    --override model.name=gpt-3.5-turbo \
    --override evaluator.batch_size=10

# Run with custom seed
python run.py configs/experiments/your_experiment.yaml --seed 123

# Run with verbose logging
python run.py configs/experiments/your_experiment.yaml --verbose
```

## Next Steps

- See `GUIDE.md` for detailed conversion from notebook
- See `ARCHITECTURE.md` for design details
- See `README.md` for full documentation
