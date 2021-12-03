from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PersonItem:
    entity_id: str
    full_name: str
    image_url: str


class PersonDB:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def exists(self, entity_id: str) -> bool:
        ...

    def save_person(self, person: PersonItem, encoding: np.ndarray) -> None:
        ...

    def find(self, encoding: np.ndarray) -> Optional[PersonItem]:
        ...
