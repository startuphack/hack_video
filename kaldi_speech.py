import json
import wave
import subprocess
from vosk import Model, KaldiRecognizer, SpkModel

SAMPLE_RATE = 16000


def wave_source(wav_file, chunk_size=SAMPLE_RATE * 4):
    wf = wave.open(wav_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be WAV format mono PCM.")

    if wf.getframerate() != SAMPLE_RATE:
        raise ValueError(f'frame rate should be {SAMPLE_RATE}')

    while True:
        data = wf.readframes(4000)
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


class KaldiProcessor:
    def __init__(self, text_models_pathes, speaker_model_path):
        self.text_models_pathes = text_models_pathes
        self.speaker_model_path = speaker_model_path

        self.text_models = list()
        for mdl_idx, model_path in enumerate(self.text_models_pathes):
            model = Model(model_path)
            rec = KaldiRecognizer(model, SAMPLE_RATE)
            rec.SetWords(True)
            if mdl_idx == 0:  # это первая, основная модель, англицизмы будем искать другими моделями
                spk_model = SpkModel(self.speaker_model_path)
                rec.SetSpkModel(spk_model)

            self.text_models_pathes.append(rec)

    def _do_process(self, frame_source):
        result = [list() for _ in self.text_models_pathes]
        for data in frame_source:
            for mdl_idx, rec in enumerate(self.text_models):
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    print("Text:", res['text'])
                    print(res)
                    if 'spk' in res:
                        print("X-vector:", res['spk'])
                        print("Speaker distance:", cosine_dist(spk_sig, res['spk']), "based on", res['spk_frames'],
                              "frames")
                        all_speakers.append(res['spk'])
                    print(json.loads(rec.Result()))
