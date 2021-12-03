from tqdm import tqdm
import pandas as pd
import tensorflow_hub as hub

import tensorflow_hub as hub
import numpy as np
import tensorflow_text

MODEL_URL = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'  # многоязычный ембеддинг
EMBEDDER = hub.load(MODEL_URL)


class SlidingEmbedder:
    def __init__(self, embedder=EMBEDDER, agg_delay=120):
        self.agg_delay = agg_delay
        self.embedder = embedder

    def make_embeddings(self, seconds, text_entries):
        agg_df = pd.DataFrame({
            'seconds': seconds,
            'texts': text_entries,
        })

        min_secs = agg_df.seconds.min()
        max_secs = agg_df.seconds.max()

        agg_results = list()

        for mid_point in tqdm(np.arange(min_secs + self.agg_delay / 2, max_secs - self.agg_delay / 2, self.agg_delay)):
            agg_data = agg_df[
                (agg_df.seconds <= mid_point + self.agg_delay / 2)
                & (agg_df.seconds >= mid_point - self.agg_delay / 2)
                ]
            if len(agg_data) > 0:
                concatenated_values = ' '.join(agg_data.texts)
                values_vector = self.embedder(concatenated_values)
                agg_results.append({
                    'mid-point': mid_point,
                    'vector': values_vector,
                })

        return agg_results
