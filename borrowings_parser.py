import pandas as pd
import json
from tqdm import tqdm
import itertools as it
import sys


borrowings = dict()
trademarks = set()
pos_sets = {'en', 'de', 'fr'}

with open('kaikki.org-dictionary-Russian.json') as json_rows:
    for line in tqdm(it.islice(json_rows.readlines(), sys.maxsize)):
        source_data = json.loads(line.strip())
        etymology_templates = source_data.get('etymology_templates', [])
        for etymology in etymology_templates:
            e_args = etymology['args']
            if '1' in e_args and '2' in e_args and '3' in e_args:
                if e_args['1'] == 'ru' and e_args['2'] in pos_sets:
                    borrowings[e_args['3']] = {
                        'word': source_data['word'],
                        'lang': e_args['2'],
                    }
#                     print(source_data['word'], e_args)

with open('borrowings.json', 'w', encoding='utf-8') as borrowings_stream:
    json.dump(borrowings, borrowings_stream, ensure_ascii=False, indent=4)

# print(len(borrowings))