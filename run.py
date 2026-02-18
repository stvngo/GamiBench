#!/usr/bin/env python3
"""
Main entry point for the research pipeline.

Supports both config-driven and CLI argument approaches.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from pipeline import Pipeline
from utils.logger import setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Config-driven approach
  python run.py configs/experiments/baseline.yaml
  
  # With overrides
  python run.py configs/experiments/baseline.yaml --override model.name=gpt-4
  
  # CLI-driven approach (if benchmark supports it)
  python run.py --benchmark tacalign --model gpt-4 --config configs/models/openai.yaml
        """
    )
    
    # Config file approach (Option C)
    parser.add_argument(
        'config',
        nargs='?',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    # CLI approach (Option A)
    parser.add_argument(
        '-b',
        '--benchmark',
        type=str,
        help='Benchmark name (alternative to config file)'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='Model name/type (alternative to config file)'
    )
    parser.add_argument(
        '-mc',
        '--model-config',
        type=str,
        help='Path to model configuration file'
    )
    
    # Overrides
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        dest='overrides',
        help='Configuration overrides in key=value format (can be used multiple times)'
    )
    
    # Other options
    parser.add_argument(
        '-od',
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        help='Override random seed'
    )
    parser.add_argument(
        '-r',
        '--resume',
        action='store_true',
        help='Resume from checkpoint (if supported)'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def parse_overrides(overrides: list) -> Dict[str, Any]:
    """
    Parse override arguments into nested dictionary.
    
    Supports dot notation: model.name=gpt-4 -> {'model': {'name': 'gpt-4'}}
    """
    if not overrides:
        return {}
    
    result = {}
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        
        key, value = override.split('=', 1)
        keys = key.split('.')
        
        # Convert value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        # Build nested dict
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return result


def create_config_from_args(args) -> str:
    """
    Create a temporary config file from CLI arguments.
    
    This allows CLI-driven approach while still using config system.
    """
    import tempfile
    import yaml
    
    config = {
        'experiment_name': f"{args.benchmark}_{args.model}",
        'seed': args.seed or 42,
        'model': {
            'type': args.model,
        },
        'dataset': {
            'path': f"data/{args.benchmark}.json",
        },
        'evaluator': {
            'type': args.benchmark,
        },
        'output_dir': args.output_dir or 'outputs/results'
    }
    
    # Load model config if provided
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
            config['model'].update(model_config)
    
    # Create temporary config file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_file)
    temp_file.close()
    
    return temp_file.name


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger(level=log_level)
    
    try:
        # Determine config path
        if args.config:
            # Config-driven approach
            config_path = args.config
        elif args.benchmark and args.model:
            # CLI-driven approach - create temp config
            config_path = create_config_from_args(args)
            logger.info(f"Created temporary config from CLI args: {config_path}")
        else:
            logger.error("Must provide either config file or --benchmark + --model")
            sys.exit(1)
        
        # Parse overrides
        overrides = parse_overrides(args.overrides or [])
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        if args.seed:
            overrides['seed'] = args.seed
        
        # Initialize and run pipeline
        pipeline = Pipeline(config_path, overrides=overrides)
        results = pipeline.run(resume=args.resume)
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
