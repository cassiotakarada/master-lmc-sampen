from dataclasses import dataclass
from typing import List


@dataclass
class Sample:
    id: int
    image_path: str
    label: int
    split: str


def rows_to_samples(rows: List[tuple]) -> List[Sample]:
    return [Sample(id=row[0], image_path=row[1], label=row[2], split=row[3]) for row in rows]
