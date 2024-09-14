import spacy
import numpy as np
from Tesi_ASDJ.src.load_file import load_file_jsonl, load_file_txt, load_directory_txt
import os
from scipy.sparse import lil_matrix

# Carico il modello di Spacy (lingua inglese small)
nlp = spacy.load("en_core_web_sm")


def one_hot_encoder(file):
    # Carico il testo dal file jsonl
    text_raw = load_file_jsonl(file)

    # Preprocessamento con Spacy, ricevo in output dei token corrispondenti alle parole
    doc = nlp(text_raw)

    # Filtro i token, rimuovo le stop words e la punteggiatura
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    # Creo il vocabolario ordinato, prendendo ogni token una sola volta (set)
    vocabulary = sorted(set(filtered_tokens))

    # Inizializo la matrice one-hot
    # (inizializzo come una matrice nulla)
    matrix = np.zeros((len(filtered_tokens), len(vocabulary)))

    # Popolo la matrice
    for i, token in enumerate(filtered_tokens):
        token_index = vocabulary.index(token)
        matrix[i, token_index] = 1

    return matrix, vocabulary


#esempio
#text_path = '../../Include/data/movies/movies_union_human_perf.jsonl'
#text_ent, vocab = one_hot_encoder(text_path)
#print(vocab)
#print(text_ent)


def one_hot_encoder_txt(text_raw):
    doc = nlp(text_raw)

    filtered_tokens = [token.text for token in doc if not token.is_punct and not token.is_stop]
    vocabulary = sorted(set(filtered_tokens))

    matrix = np.zeros((len(filtered_tokens), len(vocabulary)))
    for i, token in enumerate(filtered_tokens):
        token_index = vocabulary.index(token)
        matrix[i, token_index] = 1
    print("lunghezza vocabulary: " + str(len(vocabulary)))
    print("lunghezza filtered tokens: " + str(len(filtered_tokens)))

    for i, word in enumerate(vocabulary):
        print(f'Parola: {word}, Vettore: {matrix[i]}')

    return matrix, vocabulary


def one_hot_encoder_large_txt(text_raw):
    #Divido la stringa di input in chunk di lunghezza 500k caratteri
    chunk_size = 500000
    chunks = [text_raw[i:i + chunk_size] for i in range(0, len(text_raw), chunk_size)]

    #Inizializzo la lista dei token filtrati
    filtered_tokens = []

    #Processo un chunk alla volta e aggiungo volta per volta i token filtrati alla lista
    for chunk in chunks:
        doc = nlp(chunk)
        filtered_tokens_chunk = [token.text for token in doc if not token.is_punct and not token.is_stop]
        filtered_tokens.extend(filtered_tokens_chunk)

    #Creo un vocabolario ordinando e prendendo una sola volta i token filtrati
    vocabulary = sorted(set(filtered_tokens))

    #Inizializzo una matrice sparsa
    matrix = lil_matrix((len(filtered_tokens), len(vocabulary)))
    for i, token in enumerate(filtered_tokens):
        token_index = vocabulary.index(token)
        matrix[i, token_index] = 1
    print(f"lunghezza vocabulary: {len(vocabulary)} ")
    print(f"lunghezza filtered tokens: {len(filtered_tokens)}")
    print(f'Parole: {vocabulary}')

    return matrix, vocabulary


def one_hot_encoder_large_txt_only_vocabulary(text_raw):
    #Divido la stringa di input in chunk di lunghezza 500k caratteri
    chunk_size = 500000
    chunks = [text_raw[i:i + chunk_size] for i in range(0, len(text_raw), chunk_size)]

    #Inizializzo la lista dei token filtrati
    filtered_tokens = []

    #Processo un chunk alla volta e aggiungo volta per volta i token filtrati alla lista
    for chunk in chunks:
        doc = nlp(chunk)
        filtered_tokens_chunk = [token.text for token in doc if not token.is_punct and not token.is_stop]
        filtered_tokens.extend(filtered_tokens_chunk)

    #Creo un vocabolario ordinando e prendendo una sola volta i token filtrati
    vocabulary = sorted(set(filtered_tokens))
    return vocabulary


def one_hot_encoder_document(directory_path):
    text_raw = load_directory_txt(directory_path)
    files = [f for f in os.listdir(directory_path) if
             os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.txt')]
    num_documents = len(files)
    vocabulary = one_hot_encoder_large_txt_only_vocabulary(text_raw)

    matrix = lil_matrix((num_documents, len(vocabulary)))

    for i, file in enumerate(files):
        text_doc = load_file_txt(os.path.join(directory_path, file))
        doc = nlp(text_doc)

        filtered_tokens_doc = [token.text for token in doc if not token.is_punct and not token.is_stop]

        for token in filtered_tokens_doc:
            if token in vocabulary:
                index_token = vocabulary.index(token)

                if isinstance(index_token, int):
                    matrix[i, index_token] += 1

    return matrix, vocabulary
