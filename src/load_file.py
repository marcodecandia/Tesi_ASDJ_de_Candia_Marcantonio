import os
import json


def load_file_jsonl(file):
    data = ""
    try:
        with open(file, 'r') as file_open:
            for (line) in file_open:
                line_str = str(json.loads(line))
                data += line_str
    except FileNotFoundError:
        print(f'File {file} non trovato')

    return data

#Esempio per vedeer se effettivamente mi restituisce una stringa
#file = load_file_jsonl('../../Include/data/movies/movies_union_human_perf.jsonl')
#print(type(file))
