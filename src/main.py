from Tesi_ASDJ.src.load_file import load_file_jsonl
from Tesi_ASDJ.src.encoder import one_hot_encoder


text_path = '../data/movies/movies_union_human_perf.jsonl'
print(f"Path del file: {text_path}")

# Carico il file (in seguito ho cambiato e lo carico direttamente nella funzione one_hot_encoder)
#text_raw = load_file_jsonl(text_path)
#print(f"Contenuto del file:\n{text_raw}")

# Eseguo la funzione one_hot_encoder
matrix, text_encoded = one_hot_encoder(text_path)

# Stampo a video
print(f"Matrice di vettori one-hot-encoded: \n {matrix}")
print(f"Token codificati: \n {text_encoded}")




