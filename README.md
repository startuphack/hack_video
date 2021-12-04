# Модуль извлечения данных из видеороликов

### Установка
```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Использование
```bash
usage: main.py [-h] [--input-file INPUT_FILE] [--text-embeddings TEXT_EMBEDDINGS]
               [--find-peoples FIND_PEOPLES] [--sentiment SENTIMENT] [--summarize SUMMARIZE]
               [--verbose VERBOSE] [--max-length MAX_LENGTH]

optional arguments:
  -h, --help            show this help message and exit
  --input-file INPUT_FILE
                        input mp4 file
  --text-embeddings TEXT_EMBEDDINGS
                        add text embeddings to layers
  --find-peoples FIND_PEOPLES
                        find persons in video stream
  --sentiment SENTIMENT
                        add sentiment analysis to layers
  --summarize SUMMARIZE
                        add sber-gpt3 summarization layer
  --verbose VERBOSE     verbosity to video parsing
  --max-length MAX_LENGTH
                        maximum parsing time in seconds
```
