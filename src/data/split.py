import random
from typing import List, Dict, Tuple


def train_val_split(samples: List[Dict], val_fraction: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    if val_fraction <= 0 or val_fraction >= 1:
        return samples, []
    shuffled = samples.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]
