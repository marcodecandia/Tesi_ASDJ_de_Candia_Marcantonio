import dice_ml
import torch
import pickle
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from encoder import one_hot_encoder_document
from load_file import load_file_jsonl
import time

start_time = time.time()

# Carico il vocabolario globale
vocabulary_path = '../data/movies/vocabulary/vocabulary.txt'

with open(vocabulary_path, 'r') as f:
    vocabulary_global = [line.strip() for line in f.readlines()]

vocabulary_len = len(vocabulary_global)
print(vocabulary_len)

# Inizializzo il modello
model = NeuralNetwork(vocabulary_len)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# Filtro le feature per escludere "outcome"
#vocabulary_global = [feature for feature in vocabulary_global if feature != 'outcome']

# Carica matrix_train e matrix_test da file pickle
with open("../data/movies/matrix_train.pkl", "rb") as f:
    matrix_train = pickle.load(f)

with open("../data/movies/matrix_test.pkl", "rb") as f:
    matrix_test = pickle.load(f)

print("matrix_train e matrix_test caricati da pickle.")


# Wrapper per il modello per adattarlo all'interfaccia di Dice
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            preds = self.model(x_tensor)
            return preds.numpy()


# Definisco le feature continue e categoriali
continuous_features = vocabulary_global
print(len(
    continuous_features
))

categorical_features = []

# Carico le etichette (outcome)
train_labels = load_file_jsonl("../data/movies/train.jsonl")


# Carico documenti e trasformo in matrici di frequenze
#matrix_train, _ = one_hot_encoder_document("../data/movies/train_docs", vocabulary_global)
#matrix_test, _ = one_hot_encoder_document("../data/movies/test_docs", vocabulary_global)

array_test = [matrix_test[0].todense()]

print(matrix_train.shape)
print(matrix_test.shape)
print([matrix_test[0].todense()])
#print(continuous_features.shape)

# Assicuro che il numero di etichette corrisponda ai documenti
train_labels = train_labels[:matrix_train.shape[0]]

# Creo un DataFrame dei dati di training
data_df = pd.DataFrame(matrix_train.todense(), columns=continuous_features)

# Aggiungo la colonna target "outcome"
data_df["_outcome_"] = np.array(train_labels, dtype=np.float64)
print(f"Numero di colonne nel dataframe con l\'outcome: {len(data_df.columns)}")
#print(f"DataFrame columns: {data_df.columns.tolist()}")  # Stampa le colonne


# Verifico se tutte le feature sono presenti come colonne nel DataFrame
missing_features = set(continuous_features) - set(data_df.columns.tolist())
if missing_features:
    print(f"Le seguenti feature sono nel vocabolario ma non nel DataFrame: {missing_features}")
    # Filtro le feature mancanti
    continuous_features = [f for f in continuous_features if f in data_df.columns.tolist()]
    print(f"Le nuove feature continue aggiornate: {continuous_features}")
else:
    print("Tutte le feature sono correttamente presenti nel DataFrame.")


data = dice_ml.Data(dataframe=data_df,
                    continuous_features=continuous_features,
                    categorical_features=categorical_features,
                    outcome_name="_outcome_")

print("a")

# Definisco modello per DiCE
dice_model = dice_ml.Model(model=ModelWrapper(model), backend="PYT")

print("b")

# Seleziono una istanza di test
instance_array = np.squeeze(matrix_test[0].todense())
print(f"Shape dell'istance array dopo squeeze: {instance_array.shape}")

# Creazione del DataFrame con le colonne corrette
instance = pd.DataFrame(instance_array, columns=continuous_features)

print("c")

# Ottengo la predizione per l'istanza selezionata
prediction = ModelWrapper(model).predict(instance)
print(f"Prediction for the test instance: {prediction}")

# Inizializzo il generatore di controfattuali
exp = dice_ml.Dice(data, dice_model, method="random")

# Genero controfattuali per la prima istanza
counterfactuals_1 = exp.generate_counterfactuals(instance, total_CFs=1, desired_class="opposite")

# Visualizzo i controfattuali
counterfactuals_1.visualize_as_dataframe()

end_time = time.time()
print(f"Tempo di esecuzione: {end_time - start_time} secondi")

def dice_counterfactuals(model, vocabulary_global, matrix_train, matrix_test):
    class ModelWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, x):
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                preds = self.model(x_tensor)
                return preds.numpy()

    # Definisco le feature continue e categoriali
    continuous_features = vocabulary_global
    print(len(
        continuous_features
    ))

    categorical_features = []

    # Carico le etichette (outcome)
    train_labels = load_file_jsonl("../data/movies/train.jsonl")

    # Creo un DataFrame dei dati di training
    data_df = pd.DataFrame(matrix_train.todense(), columns=continuous_features)

    # Aggiungo la colonna target "outcome"
    data_df["_outcome_"] = np.array(train_labels, dtype=np.float64)
    print(f"Numero di colonne nel dataframe con l\'outcome: {len(data_df.columns)}")
    # print(f"DataFrame columns: {data_df.columns.tolist()}")  # Stampa le colonne

    # Verifico se tutte le feature sono presenti come colonne nel DataFrame
    missing_features = set(continuous_features) - set(data_df.columns.tolist())
    if missing_features:
        print(f"Le seguenti feature sono nel vocabolario ma non nel DataFrame: {missing_features}")
        # Filtro le feature mancanti
        continuous_features = [f for f in continuous_features if f in data_df.columns.tolist()]
        print(f"Le nuove feature continue aggiornate: {continuous_features}")
    else:
        print("Tutte le feature sono correttamente presenti nel DataFrame.")

    data = dice_ml.Data(dataframe=data_df,
                        continuous_features=continuous_features,
                        categorical_features=categorical_features,
                        outcome_name="_outcome_")

    print("a")

    # Definisco modello per DiCE
    dice_model = dice_ml.Model(model=ModelWrapper(model), backend="PYT")

    print("b")

    # Seleziono una istanza di test
    instance_array = np.squeeze(matrix_test[0].todense())
    print(f"Shape dell'istance array dopo squeeze: {instance_array.shape}")

    # Creazione del DataFrame con le colonne corrette
    instance = pd.DataFrame(instance_array, columns=continuous_features)

    print("c")

    # Ottengo la predizione per l'istanza selezionata
    prediction = ModelWrapper(model).predict(instance)
    print(f"Prediction for the test instance: {prediction}")

    # Inizializzo il generatore di controfattuali
    exp = dice_ml.Dice(data, dice_model, method="random")

    # Genero controfattuali per la prima istanza
    counterfactuals_1 = exp.generate_counterfactuals(instance, total_CFs=1, desired_class="opposite")

    # Visualizzo i controfattuali
    counterfactuals_1.visualize_as_dataframe()

