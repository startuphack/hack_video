import asyncio
import io
import logging
from typing import List, Optional

import httpx
import numpy as np
import face_recognition

from .db import PersonItem, PersonDB


WIKIDATA_URL = httpx.URL('https://query.wikidata.org')

logger = logging.getLogger(__name__)


async def query_persons(
    client: httpx.AsyncClient,
    limit: int = 100,
    offset: int = 0,
) -> List[PersonItem]:
    '''
    Загружаем известных лиц из википедии и складываем это в бд вместе с картинками
    '''
    query = '''
        SELECT DISTINCT ?person ?personLabel ?personDescription ?image WHERE {
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
            description=binding.get('personDescription', {}).get('value', ''),
            image_url=binding['image']['value'],
        )
        for binding in resp.json()['results']['bindings']
    ]


async def fetch_image(client: httpx.AsyncClient, image_url: str) -> bytes:
    resp = await client.get(image_url, follow_redirects=True)
    return resp.content


def encode_face(image_content: bytes) -> Optional[np.ndarray]:
    image = face_recognition.load_image_file(io.BytesIO(image_content))
    locations = face_recognition.face_locations(image)
    if not locations:
        return None

    return face_recognition.face_encodings(image, locations)[0]


async def process_person(queue: asyncio.Queue, db: PersonDB, client: httpx.AsyncClient) -> None:
    while True:
        person = await queue.get()
        if person is None:
            break

        if db.exists(person.entity_id):
            continue

        image_content = await fetch_image(client, person.image_url)

        encoding = encode_face(image_content)
        if encoding is None:
            continue

        if not db.exists(person.entity_id):
            db.save_person(person, encoding=encoding)
            logging.info('Saved person %s', person.full_name)


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
    for worker in workers:
        worker.add_done_callback(lambda fut: fut.result())

    total = 0
    try:
        while True:
            while queue.qsize() > limit:
                await asyncio.sleep(1)

            persons = await query_persons(
                client,
                limit=limit,
                offset=offset,
            )
            total += len(persons)
            for person in persons:
                await queue.put(person)

            logger.info('Fetched %s persons', total)

            if len(persons) < limit:
                break

            offset += limit

        for _ in range(n_workers):
            queue.put_nowait(None)

        await asyncio.gather(*workers)

    finally:
        await client.aclose()
