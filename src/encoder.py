import spacy
import numpy as np
from Tesi_ASDJ.src.load_file import load_file_jsonl

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



