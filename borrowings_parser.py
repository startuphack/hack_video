import json
from tqdm import tqdm
import itertools as it
import sys

borrowings = dict()
trademarks = set()
pos_sets = {'en', 'de', 'fr'}
# https://ru.wiktionary.org/w/index.php?title=%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%A1%D0%BB%D0%BE%D0%B2%D0%B0_%D0%B0%D0%BD%D0%B3%D0%BB%D0%B8%D0%B9%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%BF%D1%80%D0%BE%D0%B8%D1%81%D1%85%D0%BE%D0%B6%D0%B4%D0%B5%D0%BD%D0%B8%D1%8F/ru&pagefrom=%D0%B1%D1%83%D0%BB%D1%8C%D0%B4%D0%BE%D0%B3#mw-pages
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
