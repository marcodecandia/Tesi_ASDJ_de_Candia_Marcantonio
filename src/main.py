import json
from Tesi_ASDJ.src.encoder import one_hot_encoder_document, one_hot_encoder_large_txt_only_vocabulary
from Tesi_ASDJ.src.shapley_counterfactuals import shapley_counterfactuals
from Tesi_ASDJ.src.dice_counterfactuals import dice_counterfactuals
import time
from neural_network import NeuralNetwork
from optimization_loop import train_loop, test_loop
from torch import nn
import pickle
import torch
from document_dataset import DocumentDataset
from load_file import load_file_jsonl
from torch.utils.data import DataLoader

start_time = time.time()

# Recupero le label dei documenti di train e test dai file jsonl
train_labels = load_file_jsonl('../data/movies/train.jsonl')
test_labels = load_file_jsonl('../data/movies/test.jsonl')
print(train_labels)
print(test_labels)

# Definisco un vocabulary globale di train e test
directory_path_global = '../data/movies/docs'
print("Path della directory: " + directory_path_global)

# Eseguo il one hot encoder e tengo traccia del vocabulary globale
matrix, vocabulary_global = one_hot_encoder_document(directory_path_global)
print(len(vocabulary_global))
print(matrix.shape)
# Salv il vocabolario nel file vocabulary.txt
vocabulary_file_path = "../data/movies/vocabulary/vocabulary.txt"
with open(vocabulary_file_path, "w") as f:
    for word in vocabulary_global:
        # Controlla se la parola non è vuota dopo aver rimosso gli spazi
        if word.strip():
            f.write(f"{word}\n")
print(f"Vocabolario salvato correttamente in {vocabulary_file_path}")
length = len(vocabulary_global)
print(f"Dimensione del vocabolario globale: {len(vocabulary_global)}")

# Mi costruisco la matrice con vettori di frequenza per i documenti di train
directory_path_train = '../data/movies/train_docs'
matrix_train, _ = one_hot_encoder_document(directory_path_train, vocabulary_global)

#Inizializzo il modello di rete neurale
model = NeuralNetwork(len(vocabulary_global))

#Fase di train
epochs = 10
batch_size = 16
learning_rate = 0.001
weight_decay = 1e-5

#Inizializzo la loss function
loss_fn = torch.nn.BCELoss()

# Inizializzo optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Mi costruisco la matrice con vettori di frequenza per i documenti di test
directory_path_test = '../data/movies/test_docs'
matrix_test, _ = one_hot_encoder_document(directory_path_test, vocabulary_global)

# Salva matrix_train e matrix_test usando pickle
with open("../data/movies/matrix_train.pkl", "wb") as f:
    pickle.dump(matrix_train, f)

with open("../data/movies/matrix_test.pkl", "wb") as f:
    pickle.dump(matrix_test, f)

print("matrix_train e matrix_test salvati usando pickle.")

# Creo i Dataset personalizzati
train_dataset = DocumentDataset(matrix_train, train_labels)
test_dataset = DocumentDataset(matrix_test, test_labels)

# Creo i DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f"Numero di campioni nel train_dataloader: {len(train_dataloader.dataset)}")
print(f"Numero di campioni nel test_dataloader: {len(test_dataloader.dataset)}")

# Ciclo di training e testing
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1} \n ----------------------")

    train_loop(train_dataloader, model, loss_fn, optimizer, batch_size=batch_size)

    test_loop(test_dataloader, model, loss_fn)

print("Training complete")

# Salvo il modello addestrato
try:
    torch.save(model.state_dict(), 'model.pth')
    print("Modello salvato correttamente!")
except Exception as e:
    print(f"Errore durante il salvataggio del modello: {e}")


shapley_counterfactuals(model=model,
                        vocabulary_global=vocabulary_global,
                        matrix_train=matrix_train,
                        matrix_test=matrix_test)

dice_counterfactuals(model=model,
                     vocabulary_global=vocabulary_global,
                     matrix_train=matrix_train,
                     matrix_test=matrix_test)

end_time = time.time()
exec_time = end_time - start_time
print(f'Tempo di esecuzione: {exec_time} secondi, {exec_time / 60} minuti')
