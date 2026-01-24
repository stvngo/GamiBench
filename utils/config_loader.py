"""Configuration loading and validation utilities."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """
    Load configuration from YAML or JSON file.
    
    Supports hierarchical configs (base + experiment-specific).
    Can merge with environment variables and command-line overrides.
    
    Args:
        config_path: Path to config file (YAML or JSON)
        overrides: Optional dictionary of config overrides
        
    Returns:
        OmegaConf DictConfig object
        
    Example:
        >>> config = load_config("configs/experiments/baseline.yaml")
        >>> config = load_config("configs/experiments/baseline.yaml", {"model.name": "gpt-4"})
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load base config
    if config_path.suffix in ['.yaml', '.yml']:
        base_config = OmegaConf.load(config_path)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            base_config = OmegaConf.create(json.load(f))
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Check for base config reference
    if 'base_config' in base_config:
        base_path = config_path.parent / base_config.base_config
        base_config = OmegaConf.merge(OmegaConf.load(base_path), base_config)
        del base_config['base_config']
    
    # Apply overrides
    if overrides:
        override_config = OmegaConf.create(overrides)
        base_config = OmegaConf.merge(base_config, override_config)
    
    # Resolve environment variables
    base_config = OmegaConf.create(os.path.expandvars(str(base_config)))
    
    return base_config


def validate_config(config: DictConfig, required_keys: Optional[list] = None) -> None:
    """
    Validate that required configuration keys are present.
    
    Args:
        config: Configuration object to validate
        required_keys: List of required keys (dot-separated for nested)
        
    Raises:
        ValueError: If required keys are missing
    """
    if required_keys is None:
        required_keys = ['model', 'dataset', 'evaluator', 'output_dir']
    
    missing_keys = []
    for key in required_keys:
        if not OmegaConf.select(config, key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to file for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)
