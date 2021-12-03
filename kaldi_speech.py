import json
import subprocess
import wave
from copy import deepcopy, copy
import logging

from sklearn.cluster import AgglomerativeClustering
from vosk import Model, KaldiRecognizer, SpkModel

SAMPLE_RATE = 16000


def overlaps(token1, token2):
    start1, end1 = token1['start'], token1['end']
    start2, end2 = token2['start'], token2['end']
    start = max(start1, start2)
    end = min(end1, end2)

    if end > start:
        return end - start
    else:
        return 0


def wave_source(wav_file, chunk_size=SAMPLE_RATE * 4):
    wf = wave.open(wav_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM.")

    if wf.getframerate() != SAMPLE_RATE:
        raise ValueError(f'frame rate should be {SAMPLE_RATE}')

    while True:
        data = wf.readframes(chunk_size)
        if len(data) == 0:
            break
        yield data


def ffmpeg_source(mp4_file, chunk_size=SAMPLE_RATE * 4):
    process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                                mp4_file,
                                '-ar', str(SAMPLE_RATE), '-ac', '1', '-f', 's16le', '-'],
                               stdout=subprocess.PIPE)
    while True:
        data = process.stdout.read(chunk_size)
        if len(data) == 0:
            break

        yield data


# https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%A1%D0%BB%D0%BE%D0%B2%D0%B0_%D0%B0%D0%BD%D0%B3%D0%BB%D0%B8%D0%B9%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%BF%D1%80%D0%BE%D0%B8%D1%81%D1%85%D0%BE%D0%B6%D0%B4%D0%B5%D0%BD%D0%B8%D1%8F/ru&pagefrom=%D0%B0%D0%B1%D0%B8%D0%BB%D0%B8%D1%82%D0%B0%D1%86%D0%B8%D0%BE%D0%BD%D0%BD%D1%8B%D0%B9#mw-pages
class KaldiProcessor:
    def __init__(self,
                 text_models_pathes,
                 speaker_model_path,
                 search_confidence=0.75,
                 remove_confidence=0.25,
                 replacement_confidence=0.9,
                 speakers_threshold=0.6,
                 min_speaker_frames=600,
                 replacement_overlap_percent=0.8,
                 diarization_algorithm=None,
                 borrowings='default',
                 ):
        self.text_models_pathes = text_models_pathes
        self.speaker_model_path = speaker_model_path
        self.search_confidence = search_confidence
        self.remove_confidence = remove_confidence
        self.min_speaker_frames = min_speaker_frames
        self.speakers_threshold = speakers_threshold
        self.diarization_algorithm = diarization_algorithm
        self.borrowings = borrowings
        self.replacement_confidence = replacement_confidence
        self.replacement_overlap_percent = replacement_overlap_percent

        if borrowings == 'default':
            with open('borrowings.json') as borr_stream:
                self.borrowings = json.load(borr_stream)
        elif isinstance(borrowings, dict):
            self.borrowings = borrowings

        self.borrowings = {k: v for k, v in self.borrowings.items() if len(k) > 3}

        self.text_models = list()
        for mdl_idx, model_path in enumerate(self.text_models_pathes):
            model = Model(model_path)
            rec = KaldiRecognizer(model, SAMPLE_RATE)
            rec.SetWords(True)
            if mdl_idx == 0:  # это первая, основная модель, англицизмы будем искать другими моделями
                spk_model = SpkModel(self.speaker_model_path)
                rec.SetSpkModel(spk_model)

            self.text_models.append(rec)

    def _do_find_tokens(self, frame_source):
        result = [list() for _ in self.text_models_pathes]
        speakers = list()
        for data in frame_source:
            for mdl_idx, rec in enumerate(self.text_models):
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    res_list = res['result']
                    if res_list:
                        result[mdl_idx].extend(res_list)
                        if mdl_idx == 0:  # только у первой модели есть speaker model
                            start_date = res_list[0]['start']
                            stop_date = res_list[-1]['end']
                            if 'spk' in res:
                                speaker_vector = res['spk']
                                speakers.append({
                                    'start': start_date,
                                    'end': stop_date,
                                    'vector': speaker_vector,
                                    'spk_frames': res.get('spk_frames', 0),
                                })

        return {
            'tokens': result,
            'speakers': speakers,
        }

    def merge_tokens(self, results):
        tokens = results['tokens']
        replacements = list()
        for replace_tokens in tokens[1:]:
            replacements.extend([
                token for token in replace_tokens
                if token['word'] in self.borrowings and token['conf'] > self.search_confidence
            ])

        main_line = tokens[0]

        result = list()
        for token in main_line:
            confidence = token['conf']
            if confidence >= self.search_confidence:
                result.append(token)
            elif confidence >= self.remove_confidence:  # мы недостаточно уверены в токене - пытаемся найти англицизм
                replaced = False
                possible_replacements = [
                    (replacement, overlaps(replacement, token)) for replacement in replacements
                    if overlaps(replacement, token) and replacement['conf'] > self.replacement_confidence
                ]

                if possible_replacements:
                    best_match_replacement, overlap = sorted(possible_replacements, key=lambda x: x[1], reverse=True)[0]
                    if overlap / (token['end'] - token['start']) > self.replacement_overlap_percent:
                        replacing_with = self.borrowings[best_match_replacement['word']]
                        logging.info(f'replacing {token} with {best_match_replacement} ({replacing_with})')
                        result.append(
                            {
                                'conf': best_match_replacement['conf'],
                                'end': best_match_replacement['end'],
                                'start': best_match_replacement['start'],
                                'word': replacing_with['word']
                            }
                        )
                        replaced = True

                if not replaced:
                    result.append(token)
            else:
                pass  # Мы слишком не уверены в результате - не добавляем токен

        return result

    # https://wq2012.github.io/awesome-diarization/
    # ! http://www.ifp.illinois.edu/~hning2/papers/Ning_spectral.pdf
    # https://github.com/wq2012/SpectralCluster
    def find_speakers(self, results):
        speakers = results['speakers']

        if self.diarization_algorithm:
            clustering = deepcopy(self.diarization_algorithm)
        else:
            # Используем по умолчанию метод дальнего соседа
            clustering = AgglomerativeClustering(
                n_clusters=None,
                linkage='complete',
                distance_threshold=self.speakers_threshold,
                affinity='cos'
            )

        speakers = [copy(token) for token in speakers if token['spk_frames'] >= self.min_speaker_frames]

        vectors = [token['vector'] for token in speakers]

        results = clustering.fit_predict(vectors)
        for token, speaker_id in zip(speakers, results):
            token['speaker-id'] = speaker_id

        return speakers
