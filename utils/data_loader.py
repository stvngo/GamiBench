"""Data loading utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd


class DataLoader:
    """Generic data loader for various dataset formats."""
    
    @staticmethod
    def load(
        dataset_path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Load dataset from file.
        
        Supports JSON, JSONL, CSV, TSV, and the raw GamiBench folder format.
        
        Args:
            dataset_path: Path to dataset file
            format: File format ('json', 'jsonl', 'csv', 'tsv'). Auto-detected if None
            **kwargs: Additional arguments for pandas.read_csv or json.load
            
        Returns:
            Loaded data as list of dicts or DataFrame
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Auto-detect format if not specified
        if format is None:
            if dataset_path.is_dir():
                format = 'gamibench'
            else:
                suffix = dataset_path.suffix.lower()
                if suffix == '.json':
                    format = 'json'
                elif suffix == '.jsonl':
                    format = 'jsonl'
                elif suffix == '.csv':
                    format = 'csv'
                elif suffix == '.tsv':
                    format = 'tsv'
                else:
                    raise ValueError(f"Could not auto-detect format for {dataset_path}")

        format = format.lower()
        
        # Load based on format
        if format == 'json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif format == 'jsonl':
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
        elif format == 'csv':
            data = pd.read_csv(dataset_path, **kwargs)
        elif format == 'tsv':
            data = pd.read_csv(dataset_path, sep='\t', **kwargs)
        elif format == 'gamibench':
            from benchmarks.gamibench.discovery import discover_examples

            examples = discover_examples(dataset_path)
            data = [
                {
                    'name': ex.name,
                    'folder': ex.folder,
                    'normal_cp': ex.normal_cp,
                    'impossible_cp': ex.impossible_cp,
                    'viewpoints': dict(ex.viewpoints),
                }
                for ex in examples
            ]
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return data
    
    @staticmethod
    def save(
        data: Union[List[Dict], pd.DataFrame],
        output_path: Union[str, Path],
        format: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save data to file.
        
        Args:
            data: Data to save
            output_path: Output file path
            format: File format (auto-detected from extension if None)
            **kwargs: Additional arguments for pandas.to_csv or json.dump
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format
        if format is None:
            suffix = output_path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix == '.jsonl':
                format = 'jsonl'
            elif suffix == '.csv':
                format = 'csv'
            elif suffix == '.tsv':
                format = 'tsv'
            else:
                raise ValueError(f"Could not auto-detect format for {output_path}")
        
        # Save based on format
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, **kwargs)
        elif format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format == 'csv':
            data.to_csv(output_path, index=False, **kwargs)
        elif format == 'tsv':
            data.to_csv(output_path, sep='\t', index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
