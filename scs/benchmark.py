"""
Benchmark parser and validator for SCS.

Provides functions to load, validate, and parse benchmark JSON files
according to the SCS Benchmark Schema.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import glob
from datetime import datetime


@dataclass
class BenchmarkRecord:
    """
    A validated benchmark record conforming to SCS Benchmark Schema.
    """

    system: str
    config_id: str
    timestamp: Union[str, float]
    config: Dict[str, Any]
    metrics: Dict[str, float]  # {psi, rho, omega}
    raw_results: Optional[Dict[str, Any]] = None
    aux: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system': self.system,
            'config_id': self.config_id,
            'timestamp': self.timestamp,
            'config': self.config,
            'metrics': self.metrics,
            'raw_results': self.raw_results or {},
            'aux': self.aux or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkRecord':
        """Create from dictionary."""
        return cls(
            system=data['system'],
            config_id=data['config_id'],
            timestamp=data['timestamp'],
            config=data['config'],
            metrics=data['metrics'],
            raw_results=data.get('raw_results'),
            aux=data.get('aux'),
        )


class BenchmarkValidationError(Exception):
    """Raised when benchmark validation fails."""
    pass


def validate_benchmark(record: Dict[str, Any], strict: bool = True) -> bool:
    """
    Validate a benchmark record against the SCS schema.

    Args:
        record: Benchmark record dictionary
        strict: If True, raise exception on validation failure

    Returns:
        True if valid

    Raises:
        BenchmarkValidationError: If validation fails and strict=True
    """
    errors = []

    # Required fields
    required_fields = ['system', 'config_id', 'timestamp', 'config', 'metrics']
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    if errors and strict:
        raise BenchmarkValidationError(f"Validation errors: {errors}")
    elif errors:
        return False

    # Type validation
    if not isinstance(record['system'], str) or not record['system']:
        errors.append("'system' must be non-empty string")

    if not isinstance(record['config_id'], str) or not record['config_id']:
        errors.append("'config_id' must be non-empty string")

    if not isinstance(record['timestamp'], (str, int, float)):
        errors.append("'timestamp' must be string or number")

    if not isinstance(record['config'], dict):
        errors.append("'config' must be object/dict")
    elif 'algorithm' not in record['config']:
        errors.append("'config' must contain 'algorithm' field")

    # Metrics validation
    if not isinstance(record['metrics'], dict):
        errors.append("'metrics' must be object/dict")
    else:
        required_metrics = ['psi', 'rho', 'omega']
        for metric in required_metrics:
            if metric not in record['metrics']:
                errors.append(f"'metrics' missing required field: {metric}")
            else:
                value = record['metrics'][metric]
                if not isinstance(value, (int, float)):
                    errors.append(f"'metrics.{metric}' must be number")
                elif not 0.0 <= value <= 1.0:
                    errors.append(f"'metrics.{metric}' must be in range [0, 1], got {value}")

    # Optional fields type validation
    if 'raw_results' in record and not isinstance(record['raw_results'], dict):
        errors.append("'raw_results' must be object/dict if present")

    if 'aux' in record and not isinstance(record['aux'], dict):
        errors.append("'aux' must be object/dict if present")

    # Config parameter validation (if present)
    config = record.get('config', {})
    if 'ansatz_depth' in config:
        depth = config['ansatz_depth']
        if not isinstance(depth, int) or not 1 <= depth <= 10:
            errors.append(f"'config.ansatz_depth' must be integer in [1, 10], got {depth}")

    if 'learning_rate' in config:
        lr = config['learning_rate']
        if not isinstance(lr, (int, float)) or not 0.0 < lr <= 1.0:
            errors.append(f"'config.learning_rate' must be in (0, 1], got {lr}")

    if 'max_iterations' in config:
        max_iter = config['max_iterations']
        if not isinstance(max_iter, int) or max_iter < 1:
            errors.append(f"'config.max_iterations' must be positive integer, got {max_iter}")

    if errors:
        if strict:
            raise BenchmarkValidationError(f"Validation errors: {errors}")
        return False

    return True


def load_benchmark(path: Union[str, Path]) -> BenchmarkRecord:
    """
    Load and validate a single benchmark record from JSON file.

    Args:
        path: Path to benchmark JSON file

    Returns:
        BenchmarkRecord instance

    Raises:
        BenchmarkValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is malformed
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Validate
    validate_benchmark(data, strict=True)

    # Create record
    return BenchmarkRecord.from_dict(data)


def load_benchmark_batch(path: Union[str, Path]) -> List[BenchmarkRecord]:
    """
    Load a batch of benchmark records from JSON file.

    Args:
        path: Path to batch JSON file

    Returns:
        List of BenchmarkRecord instances

    Raises:
        BenchmarkValidationError: If validation fails
    """
    with open(path, 'r') as f:
        data = json.load(f)

    if 'benchmarks' not in data:
        raise BenchmarkValidationError("Batch file must contain 'benchmarks' array")

    records = []
    for i, benchmark in enumerate(data['benchmarks']):
        try:
            validate_benchmark(benchmark, strict=True)
            records.append(BenchmarkRecord.from_dict(benchmark))
        except BenchmarkValidationError as e:
            raise BenchmarkValidationError(f"Validation failed for benchmark {i}: {e}")

    return records


def load_benchmarks(path_or_pattern: Union[str, Path]) -> List[BenchmarkRecord]:
    """
    Load benchmark records from file(s) matching path or glob pattern.

    Args:
        path_or_pattern: Path to file, directory, or glob pattern

    Returns:
        List of BenchmarkRecord instances

    Examples:
        load_benchmarks("benchmarks/vqe_run_42.json")
        load_benchmarks("benchmarks/")
        load_benchmarks("benchmarks/vqe_*.json")
        load_benchmarks("benchmarks/**/*.json")
    """
    path_str = str(path_or_pattern)
    records = []

    # If it's a directory, search for all JSON files
    if Path(path_str).is_dir():
        pattern = str(Path(path_str) / "**/*.json")
        files = glob.glob(pattern, recursive=True)
    # If it contains glob characters, use glob
    elif '*' in path_str or '?' in path_str:
        files = glob.glob(path_str, recursive=True)
    # Otherwise, treat as single file
    else:
        files = [path_str]

    for file_path in files:
        try:
            # Try loading as single benchmark
            record = load_benchmark(file_path)
            records.append(record)
        except (BenchmarkValidationError, KeyError):
            # Try loading as batch
            try:
                batch_records = load_benchmark_batch(file_path)
                records.extend(batch_records)
            except (BenchmarkValidationError, KeyError):
                # Skip files that don't match schema
                pass

    return records


def write_benchmark(
    system: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    raw_results: Optional[Dict[str, Any]] = None,
    aux: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
    config_id: Optional[str] = None,
) -> str:
    """
    Write a benchmark record to JSON file.

    Args:
        system: System identifier (e.g., "vqe", "qaoa_maxcut")
        config: Configuration dictionary
        metrics: Performance triplet {psi, rho, omega}
        raw_results: Optional raw results dictionary
        aux: Optional auxiliary metadata
        output_path: Output file path (auto-generated if None)
        config_id: Configuration ID (auto-generated if None)

    Returns:
        Path to written file

    Raises:
        BenchmarkValidationError: If validation fails
    """
    # Generate config_id if not provided
    if config_id is None:
        config_id = generate_config_id(config)

    # Create benchmark record
    benchmark = {
        'system': system,
        'config_id': config_id,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics,
        'raw_results': raw_results or {},
        'aux': aux or {},
    }

    # Validate before writing
    validate_benchmark(benchmark, strict=True)

    # Generate output path if not provided
    if output_path is None:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"benchmarks/{system}_{timestamp_str}_{config_id}.json"

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)

    return str(output_path)


def generate_config_id(config: Dict[str, Any], max_length: int = 64) -> str:
    """
    Generate a configuration ID from config parameters.

    Args:
        config: Configuration dictionary
        max_length: Maximum length of generated ID

    Returns:
        Configuration ID string

    Example:
        {"algorithm": "VQE", "ansatz_type": "Metatron", "ansatz_depth": 2}
        → "vqe_metatron_d2"
    """
    parts = []

    # Algorithm (lowercase)
    if 'algorithm' in config:
        parts.append(config['algorithm'].lower())

    # Ansatz type (lowercase)
    if 'ansatz_type' in config:
        ansatz = config['ansatz_type'].lower().replace('efficient', 'eff')
        parts.append(ansatz)

    # Ansatz depth
    if 'ansatz_depth' in config:
        parts.append(f"d{config['ansatz_depth']}")

    # Optimizer (lowercase)
    if 'optimizer' in config:
        opt = config['optimizer'].lower()
        parts.append(opt)

    # Learning rate (formatted)
    if 'learning_rate' in config:
        lr = config['learning_rate']
        lr_str = f"lr{lr:.0e}".replace('-0', '').replace('e', '')
        parts.append(lr_str)

    # QAOA depth
    if 'depth' in config and config.get('algorithm') == 'QAOA':
        parts.append(f"p{config['depth']}")

    # Join parts
    config_id = '_'.join(parts)

    # Truncate if too long
    if len(config_id) > max_length:
        config_id = config_id[:max_length]

    # If empty, use hash
    if not config_id:
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        config_id = f"config_{config_hash}"

    return config_id


def filter_benchmarks(
    records: List[BenchmarkRecord],
    system: Optional[str] = None,
    algorithm: Optional[str] = None,
    min_psi: Optional[float] = None,
    min_rho: Optional[float] = None,
    min_omega: Optional[float] = None,
) -> List[BenchmarkRecord]:
    """
    Filter benchmark records by criteria.

    Args:
        records: List of BenchmarkRecord instances
        system: Filter by system identifier
        algorithm: Filter by algorithm name
        min_psi: Minimum quality threshold
        min_rho: Minimum stability threshold
        min_omega: Minimum efficiency threshold

    Returns:
        Filtered list of records
    """
    filtered = records

    if system is not None:
        filtered = [r for r in filtered if r.system == system]

    if algorithm is not None:
        filtered = [r for r in filtered if r.config.get('algorithm') == algorithm]

    if min_psi is not None:
        filtered = [r for r in filtered if r.metrics['psi'] >= min_psi]

    if min_rho is not None:
        filtered = [r for r in filtered if r.metrics['rho'] >= min_rho]

    if min_omega is not None:
        filtered = [r for r in filtered if r.metrics['omega'] >= min_omega]

    return filtered


def aggregate_benchmarks(records: List[BenchmarkRecord]) -> Dict[str, Any]:
    """
    Aggregate statistics from multiple benchmark records.

    Args:
        records: List of BenchmarkRecord instances

    Returns:
        Dictionary with aggregate statistics
    """
    if not records:
        return {}

    import numpy as np

    psi_values = [r.metrics['psi'] for r in records]
    rho_values = [r.metrics['rho'] for r in records]
    omega_values = [r.metrics['omega'] for r in records]

    return {
        'count': len(records),
        'systems': list(set(r.system for r in records)),
        'metrics': {
            'psi': {
                'mean': np.mean(psi_values),
                'std': np.std(psi_values),
                'min': np.min(psi_values),
                'max': np.max(psi_values),
            },
            'rho': {
                'mean': np.mean(rho_values),
                'std': np.std(rho_values),
                'min': np.min(rho_values),
                'max': np.max(rho_values),
            },
            'omega': {
                'mean': np.mean(omega_values),
                'std': np.std(omega_values),
                'min': np.min(omega_values),
                'max': np.max(omega_values),
            },
        },
    }
