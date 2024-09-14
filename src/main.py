from Tesi_ASDJ.src.load_file import load_file_jsonl
from Tesi_ASDJ.src.encoder import one_hot_encoder, one_hot_encoder_txt, one_hot_encoder_large_txt, one_hot_encoder_document
from Tesi_ASDJ.src.load_file import load_directory_txt, load_file_txt
import time



#text_path = '../data/movies/movies_union_human_perf.jsonl'
#print(f"Path del file: {text_path}")

# Carico il file (in seguito ho cambiato e lo carico direttamente nella funzione one_hot_encoder)
#text_raw = load_file_jsonl(text_path)
#print(f"Contenuto del file:\n{text_raw}")

# Eseguo la funzione one_hot_encoder
#matrix, text_encoded = one_hot_encoder(text_path)

# Stampo a video
#print(f"Matrice di vettori one-hot-encoded: \n {matrix}")
#print(f"Token codificati: \n {text_encoded}")


#directory_path = '../data/movies/docs'
#print("Path della directory: "+directory_path)

#text_raw = load_directory_txt(directory_path)
#print("Testo: "+ text_raw)

#matrix, text_encoded = one_hot_encoder_txt(text_raw[0:999])
#print(f'Matrice di vettori one hot encoded:  \n {matrix}')

start_time = time.time()

directory_path = '../data/movies/docs'
#Stampo path della directory di input
print("Path della directory: "+directory_path)

#Converto il contenuot dei file .txt in un'unica stringa (text_raw)
#text_raw = load_directory_txt(directory_path)
#print("Testo: "+ text_raw)

#A partire dalla string costruisco una matrice one hot encoded
#matrix, text_encoded = one_hot_encoder_large_txt(text_raw)
#print(f'Matrice di vettori one hot encoded:  \n {matrix}')

matrix, vocabulary = one_hot_encoder_document(directory_path)
print(vocabulary)
print(matrix)


end_time = time.time()
exec_time = end_time - start_time
print(f'Tempo di esecuzione: {exec_time} secondi, {exec_time/60} minuti')

