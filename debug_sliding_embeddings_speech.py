import pandas as pd
from kaldi_speech import wave_source, ffmpeg_source, KaldiProcessor
from text_embeddings import SlidingEmbedder
from files import pickle_dump, pickle_load
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logging.info('start')

if __name__ == '__main__':
    results = pickle_load('processed_result.gz.mdl')
    # print(results)
    merged_tokens = KaldiProcessor([], '').merge_tokens(results)
    tokens_df = pd.DataFrame(merged_tokens)

    model = SlidingEmbedder()
    embeddings = model.make_embeddings(tokens_df.start, tokens_df.word)
    for e in embeddings:
        print(e)
