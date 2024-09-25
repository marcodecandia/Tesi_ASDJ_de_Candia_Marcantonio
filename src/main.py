from Tesi_ASDJ.src.encoder import one_hot_encoder, one_hot_encoder_txt, one_hot_encoder_large_txt, \
    one_hot_encoder_document, one_hot_encoder_large_txt_only_vocabulary
import time
from neural_network import NeuralNetwork
from optimization_loop import train_loop, test_loop
from torch import nn
import torch
from document_dataset import DocumentDataset
from load_file import load_file_jsonl
from torch.utils.data import Dataset, DataLoader

start_time = time.time()

#Recupero le label dei documenti di train e test dai file jsonl
train_labels = load_file_jsonl('../data/movies/train.jsonl')
test_labels = load_file_jsonl('../data/movies/test.jsonl')
print(train_labels)

# Definisco un vocabulary globale di train e test
directory_path_global = '../data/movies/docs'
#Stampo path della directory che comprende documenti sia train che test
print("Path della directory: " + directory_path_global)

# Eseguo il one hot encoder e tengo traccia del vocabulary globale
#matrix, vocabulary = one_hot_encoder_document(directory_path_global)

#vocabulary_global = vocabulary
#print(vocabulary_global)
#print(len(vocabulary_global))
#Inizializzo il modello di rete neurale
model = NeuralNetwork(40188)

# Mi costruisco la matrice con vettori di frequenza per i documenti di train
directory_path_train = '../data/movies/train_docs'
matrix_train, vocabulary_train = one_hot_encoder_document(directory_path_train)

#Fase di train
epochs = 10
batch_size = 32
learning_rate = 0.001

#Inizializzo la loss function
loss_fn = nn.CrossEntropyLoss()

#Inizializzo optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Mi costruisco la matrice con vettori di frequenza per i documenti di test
directory_path_test = '../data/movies/test_docs'
matrix_test, vocabulary_test = one_hot_encoder_document(directory_path_test)

#Creo i Dataset personalizzati
train_dataset = DocumentDataset(matrix_train, train_labels)
test_dataset = DocumentDataset(matrix_test, test_labels)

#Creo i DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Ciclo di training e testing
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1} \n ----------------------")

    train_loop(train_dataloader, model, loss_fn, optimizer, batch_size=batch_size)

    test_loop(test_dataloader, model, loss_fn)

print("Training complete")

end_time = time.time()
exec_time = end_time - start_time
print(f'Tempo di esecuzione: {exec_time} secondi, {exec_time / 60} minuti')
