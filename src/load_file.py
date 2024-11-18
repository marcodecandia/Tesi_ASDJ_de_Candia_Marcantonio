import os
import json


def load_file_jsonl(file):

    labels = []

    try:
        with open(file, 'r') as file_open:
            for (line) in file_open:
                data = json.loads(line)
                if data['classification'] == 'POS':
                    labels.append(1)
                elif data['classification'] == 'NEG':
                    labels.append(0)
    except FileNotFoundError:
        print(f'File {file} non trovato')
    except Exception as e:
        print(f'Errore durante la lettura: {e}')

    return labels

def load_file_txt(file_path):

    try:
        with open(file_path, 'r') as file_open:
            text_str = str(file_open.read())
    except FileNotFoundError:
        print(f'File {file_path} non trovato')
    except Exception as e:
        print(f'Errore durante la lettura: {e}')

    return text_str

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
