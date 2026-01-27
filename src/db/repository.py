from abc import ABC, abstractmethod
from typing import List, Dict


class Repository(ABC):
    @abstractmethod
    def get_samples(self, split: str) -> List[Dict]:
        raise NotImplementedError

    def get_all_samples(self) -> List[Dict]:
        return self.get_samples(split="all")
