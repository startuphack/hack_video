import itertools as it
from collections import defaultdict

import numpy as np


def get_story_type(speakers):
    speaker_stats = defaultdict(list)
    for speaker_id, speaker_entries in it.groupby(speakers, lambda x: x['speaker-id']):
        speaker_entries = list(speaker_entries)
        start_speaker_time = speaker_entries[0]['start']
        stop_speaker_time = speaker_entries[-1]['end']

        speaker_delta = stop_speaker_time - start_speaker_time
        speaker_stats[speaker_id].append(speaker_delta)

    mean_times = [np.mean(stats) for stats in speaker_stats.values()]
    num_replies = [len(stats) for stats in speaker_stats.values()]
    if len(speaker_stats) == 2 and min(num_replies) > 10:
        # 2 спикера, не менее 10 реплик у каждого
        return 'Интервью'

    elif len(speaker_stats) == 1:
        # один спикер - одиночный репортаж
        return 'Одиночный репортаж'

    else:
        # сборная солянка - новостной сюжет
        return 'Новостной сюжет'


# results = pickle_load('processed_result.gz.mdl')
# # print(results)
# speakers = KaldiProcessor([], '').find_speakers(results)
#
# speaker_ids = [(token['start'], token['speaker-id']) for token in speakers]
#
# print(get_story_type(speakers))
