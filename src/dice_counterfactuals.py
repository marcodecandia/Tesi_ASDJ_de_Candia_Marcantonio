import dice_ml
import torch
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from encoder import one_hot_encoder_document
from load_file import load_file_jsonl

# Carico il vocabolario globale
vocabulary_path = '../data/movies/vocabulary/vocabulary.txt'

with open(vocabulary_path, 'r') as f:
    vocabulary_global = [line.strip() for line in f.readlines()]

vocabulary_len = len(vocabulary_global)

# Inizializzo il modello
model = NeuralNetwork(vocabulary_len)
model.load_state_dict(torch.load("Model_1.pth", weights_only=True))
model.eval()


# Wrapper per il modello per adattarlo all'inetrfaccia di Dice
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            preds = self.model(x_tensor)
            return preds.numpy()


continuous_features = vocabulary_global
categorical_features = []

train_labels = load_file_jsonl("../data/movies/train.jsonl")
print(train_labels)

# Carico documenti e trasformo in matrici di frequenze
matrix_train, _ = one_hot_encoder_document("../data/movies/train_docs", vocabulary_global)
matrix_test, _ = one_hot_encoder_document("../data/movies/test_docs", vocabulary_global)

# Creo un Dataframe dei dati di training
data_df = pd.DataFrame(matrix_train.todense(), columns=categorical_features)
data_df["outcome"] = train_labels
print(len(data_df.columns))
print(len(continuous_features))

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
                    outcome_name="outcome")

# Definisco modello per Dice
dice_model = dice_ml.Model(model=ModelWrapper(model), backend="PYT")

# Seleziono una istanza di test
instance = pd.DataFrame([matrix_test[0].todense().tolist()[0]], columns=continuous_features)

prediction = ModelWrapper(model).predict([instance])
print(prediction)

# Inizializzo generatore di controfattuali
exp = dice_ml.Dice(data, dice_model, method="random")

# Genero controfattuali per la prima istanza
counterfactuals_1 = exp.generate_counterfactuals(instance, total_CFs=3, desired_class="opposite")

counterfactuals_1.visualize_as_dataframe()
