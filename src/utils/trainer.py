from abc import ABC, abstractmethod
from typing import Dict


class BaseTrainer(ABC):
    """
    Minimal trainer interface so we can later plug in
    alternative implementations (e.g. DPTrainer) without
    touching model or loss definitions.
    """

    @abstractmethod
    def train(self, num_epochs: int) -> Dict[str, float]:
        """
        Run training for a fixed number of epochs.
        Returns a dictionary of summary metrics.
        """

    @abstractmethod
    def evaluate(self, split: str) -> Dict[str, float]:
        """
        Evaluate on a given split name ("train", "val", "test").
        """

