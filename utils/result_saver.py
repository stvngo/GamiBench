"""Utilities for saving and loading evaluation results."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import pandas as pd


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    config: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> Path:
    """
    Save evaluation results to disk.
    
    Saves:
    - results.json: Full results dictionary
    - metrics.json: Aggregated metrics
    - config.yaml: Frozen configuration (if provided)
    - metadata.json: Additional metadata (if provided)
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save results
        config: Configuration used for the experiment (optional)
        metadata: Additional metadata (timestamp, git hash, etc.)
        
    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_path = output_dir / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Extract and save metrics if available
    if 'metrics' in results:
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results['metrics'], f, indent=2, ensure_ascii=False, default=str)
    
    # Save config if provided
    if config is not None:
        try:
            from omegaconf import OmegaConf
            config_path = output_dir / "config.yaml"
            OmegaConf.save(config, config_path)
        except ImportError:
            # Fallback to JSON
            config_path = output_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    metadata['timestamp'] = datetime.now().isoformat()
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    return output_dir


def load_results(results_dir: str) -> Dict[str, Any]:
    """
    Load evaluation results from disk.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        Dictionary containing results, metrics, config, and metadata
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    loaded = {}
    
    # Load results
    results_path = results_dir / "results.json"
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            loaded['results'] = json.load(f)
    
    # Load metrics
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            loaded['metrics'] = json.load(f)
    
    # Load config
    config_path = results_dir / "config.yaml"
    if config_path.exists():
        try:
            from omegaconf import OmegaConf
            loaded['config'] = OmegaConf.load(config_path)
        except ImportError:
            pass
    
    # Fallback to JSON config
    if 'config' not in loaded:
        config_path = results_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded['config'] = json.load(f)
    
    # Load metadata
    metadata_path = results_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            loaded['metadata'] = json.load(f)
    
    return loaded
