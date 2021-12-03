import pandas as pd
from kaldi_speech import KaldiProcessor, ffmpeg_source


def process_file(mp4_file, args=None):
    MODELS_PATHES = [
        'models/vosk-model-small-ru-0.22',  # основная модель
        'models/vosk-model-small-en-us-0.15'  # модель для поиска англицизмов
    ]

    SPEAKER_MODEL = 'models/vosk-model-spk-0.4'

    processor = KaldiProcessor(MODELS_PATHES, SPEAKER_MODEL, max_len = args.max_length)

    result_layers = processor.get_layers(ffmpeg_source(mp4_file))

    ok_tokens = result_layers['text-tokens']

    tokens_df = pd.DataFrame(ok_tokens)

    if args.text_embeddings:
        from text_embeddings import SlidingEmbedder

        model = SlidingEmbedder()
        embeddings = model.make_embeddings(tokens_df.start, tokens_df.word)

        result_layers['text-embeddings'] = embeddings

    if args.find_peoples:
        from video.detections import process_file
        persons_data = process_file(mp4_file, 'video/persons.db', max_len = args.max_length)
        result_layers['persons'] = persons_data

    return result_layers
