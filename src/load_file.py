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
    except Exception as e:
        print(f'Errore durante la lettura: {e}')

    return data


#Esempio per vedeer se effettivamente mi restituisce una stringa
#file = load_file_jsonl('../../Include/data/movies/movies_union_human_perf.jsonl')
#print(type(file))

def load_directory_txt(directory_path):
    data = ""
    try:
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path) and file.endswith('.txt'):
                with open(file_path, 'r') as file_open:
                    text_file = file_open.read()
                    data += text_file
    except FileNotFoundError:
        print(f'Directory {directory_path} non trovata')
    except Exception as e:
        print(f'Errore durante la lettura: {e}')
    return data
