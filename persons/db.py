import contextlib
import sqlite3
from dataclasses import dataclass, astuple
from typing import Optional

import numpy as np


@dataclass
class PersonItem:
    entity_id: str
    full_name: str
    description: str
    image_url: str


class PersonDB:

    _create_table_q = (
        'CREATE TABLE IF NOT EXISTS person '
        '(entity_id text PRIMARY KEY, full_name text, description text, image_url text, encoding blob)'
    )
    _exists_q = (
        'SELECT 1 FROM person WHERE entity_id = ?'
    )
    _insert_q = (
        'INSERT INTO person (entity_id, full_name, description, image_url, encoding) '
        'VALUES (?, ?, ?, ?, ?)'
    )
    _load_encodings_q = (
        'SELECT entity_id, encoding FROM person'
    )
    _get_person_q = (
        'SELECT entity_id, full_name, description, image_url FROM person '
        'WHERE entity_id = :entity_id'
    )

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._encodings = None
        self._index = None

        with self._connect() as cur:
            cur.execute(self._create_table_q)

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    def load_encodings(self):
        with self._connect() as cur:
            cur.execute(self._load_encodings_q)
            rows = cur.fetchall()

        self._encodings = np.stack([np.frombuffer(raw_encoding) for _, raw_encoding in rows])
        self._index = dict(zip(range(len(rows)), (entity_id for entity_id, _ in rows)))

    def exists(self, entity_id: str) -> bool:
        with self._connect() as cur:
            cur.execute(self._exists_q, (entity_id,))
            return bool(cur.fetchone())

    def save_person(self, person: PersonItem, encoding: np.ndarray) -> None:
        with self._connect() as cur:
            cur.execute(self._insert_q, [*astuple(person), encoding])

    def get_person(self, entity_id: str) -> PersonItem:
        with self._connect() as cur:
            cur.execute(self._get_person_q, (entity_id,))
            row = cur.fetchone()

        if row is None:
            raise LookupError(entity_id)

        return PersonItem(*row)

    def find(self, encoding: np.ndarray, tolerance: float = 0.7) -> Optional[PersonItem]:
        if self._encodings is None:
            self.load_encodings()

        assert self._index
        distances = np.linalg.norm(encoding - self._encodings , axis=1)
        if distances.min() > tolerance:
            return None

        entity_id = self._index[np.argmin(distances)]
        return self.get_person(entity_id)
