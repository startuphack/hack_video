import asyncio
import pathlib
from typing import List, Optional

import httpx
import numpy as np

from .db import PersonItem, PersonDB


WIKIDATA_URL = httpx.URL('https://query.wikidata.org')


async def query_persons(
    client: httpx.AsyncClient,
    limit: int = 100,
    offset: int = 0,
) -> List[PersonItem]:
    query = '''
        SELECT ?person ?personLabel ?image WHERE {
          ?person wdt:P31 wd:Q5. # Human
          ?person wdt:P27 wd:Q159. # Russian
          ?person wdt:P18 ?image. # With image
          ?person wdt:P106 wd:Q82955 . # Items that have "occupation (P106): politician (Q82955)"
          FILTER NOT EXISTS{ ?person wdt:P570 ?date } # Living (date of death not exists)
          SERVICE wikibase:label { bd:serviceParam wikibase:language "ru". }
        } ORDER BY ?person
    '''

    resp = await client.get(
        WIKIDATA_URL.join('/sparql'),
        params={
            'query': f'{query} LIMIT {limit} OFFSET {offset}',
            'format': 'json',
        },
    )
    return [
        PersonItem(
            entity_id=binding['person']['value'],
            full_name=binding['personLabel']['value'],
            image_url=binding['image']['value'],
        )
        for binding in resp.json()['results']['bindings']
    ]


async def fetch_image(client: httpx.AsyncClient, image_url: str) -> bytes:
    resp = await client.get(image_url)
    return resp.content


def encode_face(image_content: bytes) -> Optional[np.ndarray]:
    _locations = face_recognition.face_locations(rgb_frame)
    _encodings = face_recognition.face_encodings(rgb_frame, _locations)


async def process_person(queue: asyncio.Queue, db: PersonDB, client: httpx.AsyncClient) -> None:
    while True:
        person = await queue.get()
        if person is None:
            break

        if db.exists(person.entity_id):
            return

        image_content = await fetch_image(client, person.image_url)
        encoding = encode_face(image_content)
        if encoding is None:
            return

        db.save_person(person, encoding=encoding)


async def scrap_persons(
    db_path: str,
    page_size: int = 10,
    n_workers: int = 4,
    timeout: int = 60 * 5,
) -> None:

    db = PersonDB(db_path)
    client = httpx.AsyncClient(timeout=timeout)
    queue = asyncio.Queue()

    limit = page_size
    offset = 0

    workers = [
        asyncio.create_task(process_person(queue, db, client))
        for _ in range(n_workers)
    ]
    try:
        while True:
            persons = await query_persons(
                client,
                limit=limit,
                offset=offset,
            )
            for person in persons:
                await queue.put(person)

            if len(persons) < limit:
                break

            offset += limit

        for _ in range(n_workers):
            queue.put_nowait(None)

        await asyncio.gather(*workers)

    finally:
        await client.aclose()
