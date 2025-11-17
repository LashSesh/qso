"""
Configuration representation for Q⊗DASH (Metatron VM).

A configuration c ∈ C encodes the hyperparameters, ansatz choices, and
optimizer settings for the quantum-hybrid algorithms.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import json
import copy


@dataclass
class Configuration:
    """
    Represents a configuration of Q⊗DASH.

    A configuration includes algorithm family, ansatz type, optimizer,
    and associated hyperparameters.
    """

    # Algorithm family (VQE, QAOA, QuantumWalk, Grover, Boson, VQC, etc.)
    algorithm: str = "VQE"

    # Ansatz configuration
    ansatz_type: str = "Metatron"
    ansatz_depth: int = 2

    # Optimizer configuration
    optimizer: str = "Adam"
    learning_rate: float = 0.01
    max_iterations: int = 100

    # Algorithm-specific parameters
    num_random_starts: int = 1
    dephasing_rate: Optional[float] = None
    shot_count: Optional[int] = None

    # Additional hyperparameters
    params: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    name: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize configuration to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Configuration":
        """Create configuration from dictionary."""
        # Filter out keys not in dataclass fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "Configuration":
        """Deserialize configuration from JSON."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "Configuration":
        """Load configuration from file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())

    def to_file(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def copy(self) -> "Configuration":
        """Create a deep copy of the configuration."""
        return copy.deepcopy(self)

    def distance(self, other: "Configuration") -> float:
        """
        Compute a distance metric between two configurations.

        This is a simplified metric for configuration space that considers
        discrete parameter changes and continuous parameter differences.
        """
        dist = 0.0

        # Discrete parameters (0 or 1 contribution each)
        if self.algorithm != other.algorithm:
            dist += 1.0
        if self.ansatz_type != other.ansatz_type:
            dist += 1.0
        if self.optimizer != other.optimizer:
            dist += 1.0

        # Continuous parameters (normalized differences)
        dist += abs(self.ansatz_depth - other.ansatz_depth) / 10.0
        dist += abs(self.learning_rate - other.learning_rate) / 0.1
        dist += abs(self.max_iterations - other.max_iterations) / 100.0
        dist += abs(self.num_random_starts - other.num_random_starts) / 5.0

        return dist


class ConfigurationSpace:
    """
    Manages the space of admissible configurations C.

    Provides methods for generating, validating, and navigating
    configuration space.
    """

    # Allowed values for discrete parameters
    ALGORITHMS = ["VQE", "QAOA", "QuantumWalk", "Grover", "Boson", "VQC", "Integration"]
    ANSATZ_TYPES = ["HardwareEfficient", "EfficientSU2", "Metatron"]
    OPTIMIZERS = ["Adam", "LBFGS", "GradientDescent", "COBYLA"]

    def __init__(self):
        """Initialize configuration space."""
        self.current: Optional[Configuration] = None
        self.history: List[Configuration] = []

    def default_configuration(self) -> Configuration:
        """Return a default baseline configuration."""
        return Configuration(
            algorithm="VQE",
            ansatz_type="Metatron",
            ansatz_depth=2,
            optimizer="Adam",
            learning_rate=0.01,
            max_iterations=100,
            num_random_starts=1,
            name="default",
        )

    def is_valid(self, config: Configuration) -> bool:
        """Check if a configuration is admissible."""
        if config.algorithm not in self.ALGORITHMS:
            return False
        if config.ansatz_type not in self.ANSATZ_TYPES:
            return False
        if config.optimizer not in self.OPTIMIZERS:
            return False
        if config.ansatz_depth < 1 or config.ansatz_depth > 10:
            return False
        if config.learning_rate <= 0 or config.learning_rate > 1.0:
            return False
        if config.max_iterations < 1:
            return False
        return True

    def set_current(self, config: Configuration) -> None:
        """Set the current active configuration."""
        if not self.is_valid(config):
            raise ValueError(f"Invalid configuration: {config}")
        self.current = config
        self.history.append(config.copy())

    def generate_neighbors(
        self, config: Configuration, num_neighbors: int = 5
    ) -> List[Configuration]:
        """
        Generate neighboring configurations in configuration space.

        Used by the double-kick operator to explore local variations.
        """
        neighbors = []

        for _ in range(num_neighbors):
            neighbor = config.copy()

            # Randomly perturb one parameter
            import random

            param_choice = random.choice(
                [
                    "ansatz_depth",
                    "learning_rate",
                    "max_iterations",
                    "num_random_starts",
                    "optimizer",
                    "ansatz_type",
                ]
            )

            if param_choice == "ansatz_depth":
                neighbor.ansatz_depth = max(
                    1, min(10, config.ansatz_depth + random.choice([-1, 0, 1]))
                )
            elif param_choice == "learning_rate":
                neighbor.learning_rate = max(
                    0.001, min(0.1, config.learning_rate * random.uniform(0.8, 1.2))
                )
            elif param_choice == "max_iterations":
                neighbor.max_iterations = max(
                    10,
                    min(
                        500,
                        config.max_iterations + random.choice([-20, -10, 0, 10, 20]),
                    ),
                )
            elif param_choice == "num_random_starts":
                neighbor.num_random_starts = max(
                    1, min(5, config.num_random_starts + random.choice([-1, 0, 1]))
                )
            elif param_choice == "optimizer":
                neighbor.optimizer = random.choice(self.OPTIMIZERS)
            elif param_choice == "ansatz_type":
                neighbor.ansatz_type = random.choice(self.ANSATZ_TYPES)

            if self.is_valid(neighbor):
                neighbors.append(neighbor)

        return neighbors
