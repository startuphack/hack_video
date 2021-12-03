from collections import defaultdict
import pandas as pd
from kaldi_speech import KaldiProcessor, ffmpeg_source
from ner_text import get_default_ners
from text_embeddings import SlidingEmbedder


def get_full_text(tokens_list):
    return ' '.join(token['word'] for token in tokens_list)

from video.detections import process_file
from sentiment import get_sentinent


def process_file(mp4_file, args=None):
    MODELS_PATHES = [
        'models/vosk-model-small-ru-0.22',  # основная модель
        'models/vosk-model-small-en-us-0.15'  # модель для поиска англицизмов
    ]

    SPEAKER_MODEL = 'models/vosk-model-spk-0.4'

    processor = KaldiProcessor(MODELS_PATHES, SPEAKER_MODEL, max_len=args.max_length)

    result_layers = processor.get_layers(ffmpeg_source(mp4_file))

    ok_tokens = result_layers['text-tokens']
    full_text = get_full_text(ok_tokens)
    ner_dict = get_default_ners(full_text)
    result_layers['named-entities'] = ner_dict

    tokens_df = pd.DataFrame(ok_tokens)

    if args.sentiment:
        model = SlidingEmbedder(embedder=get_sentinent)
        sentiments = model.make_embeddings(tokens_df.start, tokens_df.word)
        result_layers['sentinent'] = sentiments

    if args.text_embeddings:

        model = SlidingEmbedder()
        embeddings = model.make_embeddings(tokens_df.start, tokens_df.word)

        result_layers['text-embeddings'] = embeddings

    if args.find_peoples:
        persons_data = process_file(mp4_file, 'video/persons.db', max_len=args.max_length)
        result_layers['persons'] = persons_data

        objects_df = pd.DataFrame.from_records(
            [
                (timestamp, ' '.join(yolo_tags))
                for timestamp, yolo_tags in persons_data.objects
                if yolo_tags
            ],
            columns=['timestamp', 'yolo_tags'],
        )

        person_texts_by_ts = defaultdict(list)
        for person in persons_data.persons:
            if person.db_item is None:
                continue

            for timestamp in person.timestamps:
                tags = f'{person.db_item.full_name} {person.db_item.description}'
                person_texts_by_ts[timestamp].append(tags)

        persons_df = pd.DataFrame.from_records(
            [
                (timestamp, ' '.join(texts))
                for timestamp, texts in sorted(person_texts_by_ts.items())
            ],
            columns=['timestamp', 'person_text'],
        )

        model = SlidingEmbedder()
        person_embeddings = model.make_embeddings(objects_df.timestamp, objects_df.yolo_tags)
        objects_embeddings = model.make_embeddings(persons_df.timestamp, persons_df.person_text)
        result_layers['person-embeddings'] = person_embeddings
        result_layers['objects-embeddings'] = objects_embeddings

    return result_layers
