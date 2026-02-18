"""Main pipeline orchestration class."""

from pathlib import Path
import inspect
from typing import Any, Dict, Optional

from models.base import BaseModel
from models.model_factory import ModelFactory
from evaluators.base import BaseEvaluator
from utils.config_loader import load_config, validate_config
from utils.logger import setup_logger
from utils.seeding import set_seed
from utils.data_loader import DataLoader
from utils.result_saver import save_results


class Pipeline:
    """
    Main pipeline for running end-to-end evaluations.
    
    Handles:
    - Configuration loading and validation
    - Model initialization
    - Data loading
    - Evaluator initialization
    - Execution and result saving
    """
    
    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            overrides: Optional configuration overrides
        """
        self.config_path = Path(config_path).resolve()

        # Load and validate config
        self.config = load_config(str(self.config_path), overrides=overrides)
        validate_config(self.config)
        
        # Setup logging
        log_dir = self.config.get('log_dir', 'outputs/logs')
        self.logger = setup_logger(
            name=self.config.get('experiment_name', 'pipeline'),
            log_dir=log_dir
        )
        
        # Set seed for reproducibility
        seed = self.config.get('seed', 42)
        set_seed(seed)
        self.logger.info(f"Set random seed to {seed}")
        
        # Initialize components
        self.model = None
        self.data = None
        self.evaluator = None
        
        self.logger.info(f"Pipeline initialized with config: {self.config_path}")
    
    def _load_model(self) -> BaseModel:
        """Load and initialize the model."""
        if self.model is not None:
            return self.model
        
        self.logger.info("Loading model...")
        model_config = self.config.model
        
        # Handle API keys from environment
        if 'api_key' in model_config and isinstance(model_config.api_key, str):
            import os
            if model_config.api_key.startswith("${") and model_config.api_key.endswith("}"):
                env_var = model_config.api_key[2:-1]
                model_config.api_key = os.getenv(env_var)
                if not model_config.api_key:
                    raise ValueError(f"Environment variable {env_var} not set")
            elif model_config.api_key.startswith("$"):
                env_var = model_config.api_key[1:]
                model_config.api_key = os.getenv(env_var)
                if not model_config.api_key:
                    raise ValueError(f"Environment variable {env_var} not set")
        
        self.model = ModelFactory.create(model_config)
        self.logger.info(f"Model loaded: {self.model}")
        return self.model
    
    def _load_data(self) -> list:
        """Load the evaluation dataset."""
        if self.data is not None:
            return self.data
        
        self.logger.info("Loading data...")
        dataset_config = self.config.dataset
        dataset_path = Path(dataset_config.path)
        
        # Support relative paths from config directory and cwd
        if not dataset_path.is_absolute():
            config_relative = (self.config_path.parent / dataset_path).resolve()
            cwd_relative = (Path.cwd() / dataset_path).resolve()
            if config_relative.exists():
                dataset_path = config_relative
            else:
                dataset_path = cwd_relative
        
        self.data = DataLoader.load(dataset_path, format=dataset_config.get('format'))
        self.logger.info(f"Loaded {len(self.data)} examples")
        return self.data
    
    def _load_evaluator(self) -> BaseEvaluator:
        """Load and initialize the evaluator."""
        if self.evaluator is not None:
            return self.evaluator
        
        self.logger.info("Loading evaluator...")
        evaluator_config = self.config.evaluator
        
        # Dynamically import evaluator class
        evaluator_type = evaluator_config.type
        module_path = evaluator_config.get('module', f'evaluators.{evaluator_type}')
        class_name = evaluator_config.get('class', f'{evaluator_type.capitalize()}Evaluator')
        
        try:
            module = __import__(module_path, fromlist=[class_name])
            evaluator_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not import evaluator {class_name} from {module_path}: {e}"
            )
        
        # Initialize evaluator
        model = self._load_model()
        data = self._load_data()
        init_sig = inspect.signature(evaluator_class.__init__)
        if "full_config" in init_sig.parameters:
            self.evaluator = evaluator_class(model, data, evaluator_config, full_config=self.config)
        else:
            self.evaluator = evaluator_class(model, data, evaluator_config)
        self.logger.info(f"Evaluator loaded: {self.evaluator}")
        return self.evaluator
    
    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.
        
        Args:
            resume: Whether to resume from checkpoint (if supported)
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Starting pipeline execution...")
        
        # Load evaluator (which loads model and data)
        evaluator = self._load_evaluator()
        
        # Run evaluation
        self.logger.info("Running evaluation...")
        evaluate_sig = inspect.signature(evaluator.evaluate)
        if "resume" in evaluate_sig.parameters:
            results = evaluator.evaluate(resume=resume)
        else:
            results = evaluator.evaluate()
        
        # Save results
        output_dir = self.config.get('output_dir', 'outputs/results')
        experiment_name = self.config.get('experiment_name', 'experiment')
        timestamp = self.config.get('timestamp', None)
        
        if timestamp:
            output_path = Path(output_dir) / f"{experiment_name}_{timestamp}"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(output_dir) / f"{experiment_name}_{timestamp}"
        
        # Save frozen config with results
        save_results(
            results=results,
            output_dir=str(output_path),
            config=self.config,
            metadata={
                'experiment_name': experiment_name,
                'model': str(self.model),
                'num_examples': len(self.data)
            }
        )
        
        self.logger.info(f"Results saved to {output_path}")
        self.logger.info(f"Metrics: {results.get('metrics', {})}")
        
        return results
    
    def __repr__(self) -> str:
        return f"Pipeline(config={self.config.get('experiment_name', 'unknown')})"
