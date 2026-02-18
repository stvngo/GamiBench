# Architecture Overview

## Design Principles

1. **Modularity**: Each component (model, evaluator, data loader) is independent
2. **Reproducibility**: Configs are frozen with results, seeds are set automatically
3. **Extensibility**: Easy to add new models/evaluators via factory pattern
4. **Robustness**: Graceful error handling, resume capability
5. **Observability**: Comprehensive logging, progress tracking

## Component Overview

### Pipeline (`pipeline.py`)
- Main orchestration class
- Handles: config loading → model init → data loading → evaluator init → execution → result saving
- Entry point for all experiments

### Models (`models/`)
- **BaseModel**: Abstract interface all models must implement
- **ModelFactory**: Registry pattern for model creation
- **Concrete Models**: OpenAI, local models, etc.

### Evaluators (`evaluators/`)
- **BaseEvaluator**: Abstract interface for benchmarks
- **Concrete Evaluators**: Benchmark-specific implementations
- Handles: single example evaluation, batch processing, metric computation

### Utilities (`utils/`)
- **config_loader**: YAML/JSON config loading with validation
- **logger**: Structured logging to file and console
- **seeding**: Reproducibility via random seed setting
- **data_loader**: Generic data loading (JSON, JSONL, CSV, TSV)
- **result_saver**: Save/load results with metadata

### Configuration (`configs/`)
- Hierarchical configs (base + experiment-specific)
- Environment variable support
- Model/dataset/evaluator configs

## Data Flow

```
Config File
    ↓
Pipeline.__init__()
    ↓
Pipeline._load_model() → ModelFactory.create()
    ↓
Pipeline._load_data() → DataLoader.load()
    ↓
Pipeline._load_evaluator() → Evaluator.__init__(model, data, config)
    ↓
Pipeline.run() → Evaluator.evaluate()
    ↓
Evaluator.evaluate_batch() → [evaluate_single() for each example]
    ↓
Evaluator.compute_metrics()
    ↓
Pipeline.run() → save_results()
    ↓
outputs/results/{experiment_name}_{timestamp}/
```

## Extension Points

### Adding a Model
1. Create class inheriting `BaseModel`
2. Implement `generate()` and `score()`
3. Register in `ModelFactory`

### Adding an Evaluator
1. Create class inheriting `BaseEvaluator`
2. Implement `evaluate()` and `evaluate_single()`
3. Optionally override `compute_metrics()`

### Adding a Data Format
1. Extend `DataLoader.load()` with new format
2. Or create custom loader in your evaluator

## Error Handling Strategy

- **Model Errors**: Caught at generation time, logged, example marked as failed
- **Data Errors**: Validated at load time, clear error messages
- **Config Errors**: Validated at pipeline init, missing keys reported
- **Evaluation Errors**: Caught per-example, logged, continue with remaining examples

## Performance Considerations

- **Batch Processing**: Models can override `batch_generate()` for optimization
- **Parallelization**: Evaluators can override `evaluate_batch()` for multiprocessing
- **Caching**: Can be added at model or evaluator level
- **Progress Tracking**: Built-in via tqdm in `evaluate_batch()`

## Testing Strategy

- **Unit Tests**: Test individual components (models, evaluators, utilities)
- **Integration Tests**: Test full pipeline with mock data
- **Smoke Tests**: Quick validation with small dataset

## Future Enhancements

- Experiment tracking integration (wandb/mlflow)
- Checkpointing for long-running evaluations
- Distributed evaluation support
- Result visualization tools
- Automated report generation
