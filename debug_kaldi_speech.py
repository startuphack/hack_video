from kaldi_speech import wave_source, ffmpeg_source, KaldiProcessor
from files import pickle_dump, pickle_load
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logging.info('start')

def make_result():
    MODELS_PATHES = [
        'models/vosk-model-small-ru-0.22',  # основная модель
        'models/vosk-model-small-en-us-0.15'  # модель для поиска англицизмов
    ]

    SPEAKER_MODEL = 'models/vosk-model-spk-0.4'

    processor = KaldiProcessor(MODELS_PATHES, SPEAKER_MODEL)
    wave_path = '/media/aleksander/hd2/data/hackthehack/rbk/out.wav'

    result = processor._do_find_tokens(wave_source(wave_path))

    pickle_dump(result, 'processed_result.gz.mdl')

def debug_merge_tokens():
    results = pickle_load('processed_result.gz.mdl')
    # print(results)
    KaldiProcessor([], '').merge_tokens(results)

def debug_diarization():
    results = pickle_load('processed_result.gz.mdl')
    # print(results)
    KaldiProcessor([], '').find_speakers(results)

if __name__ == '__main__':
    debug_merge_tokens()