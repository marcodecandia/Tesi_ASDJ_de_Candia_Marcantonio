import torch
import numpy as np
import shap
import pickle
from neural_network import NeuralNetwork
from document_dataset import DocumentDataset
from load_file import load_file_jsonl
from encoder import one_hot_encoder_document

# Carico il vocabolario globale
vocabulary_path = '../data/movies/vocabulary/vocabulary.txt'

with open(vocabulary_path, 'r') as f:
    vocabulary_global = [line.strip() for line in f.readlines()]

print("Vocabolario caricato con successo")
#print(vocabulary_global)

vocab_size = len(vocabulary_global)
print(vocab_size)

# Inizializzo il modello
model = NeuralNetwork(vocab_size)

# Carico i pesi salvati nel training
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# Carico i dati di test
directory_path_test = '../data/movies/test_docs'

# Carica matrix_train e matrix_test da file pickle
with open("../data/movies/matrix_train.pkl", "rb") as f:
    matrix_train = pickle.load(f)

with open("../data/movies/matrix_test.pkl", "rb") as f:
    matrix_test = pickle.load(f)

print("matrix_train e matrix_test caricati da pickle.")

#matrix_train, _ = one_hot_encoder_document("../data/movies/train_docs", vocabulary_global)
print(f"Dimensione della matrice di addestramento: {matrix_train.shape}")
#matrix_test, _ = one_hot_encoder_document(directory_path_test, vocabulary_global)

print(f"Dimensione della matrice di test: {matrix_test.shape}")

# Calcolo i valori SHAP
explainer = shap.DeepExplainer(model, torch.tensor(matrix_train.todense(), dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(matrix_test.todense(), dtype=torch.float32), check_additivity=False)

print(f"Il valore atteso del modello rispetto al dataset è: {explainer.expected_value}")

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.squeeze(shap_values)

print(f"Shape of shap_values: {shap_values.shape}")
print(f"Shape of data used for SHAP: {matrix_test.shape}")

# Converto i valori SHAP in un oggetto Explanation
shap_values_explanation = shap.Explanation(values=shap_values,
                                           base_values=explainer.expected_value,
                                           data=matrix_test.todense(),
                                           feature_names=vocabulary_global)
print(f"explanation shape: {shap_values_explanation.shape}")

# Visualizzo i valori SHAP
try:
    shap.plots.beeswarm(shap_values_explanation)
except Exception as e:
    print(f"Errore durante la creazione del beeswarm plot: {e}")

# Visualizzazione alternativa: bar plot
try:
    # Bar plot per vedere l'importanza media delle feature
    shap.plots.bar(shap_values_explanation, max_display=100)
except Exception as e:
    print(f"Errore durante la creazione del bar plot: {e}")

try:
    shap.plots.heatmap(shap_values_explanation[:1000])
except Exception as e:
    print(f"Errore durante la creazione del beeswarm plot: {e}")

def shapley_counterfactuals(model, vocabulary_global, matrix_train, matrix_test):
    #Dimensione del vocabolario
    vocab_size = len(vocabulary_global)
    print(f"Dimensione del vocabolario globale: {vocab_size}")

    # Carico i pesi salvati nel training
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    print(f"Dimensione della matrice di addestramento: {matrix_train.shape}")
    print(f"Dimensione della matrice di test: {matrix_test.shape}")

    # Calcolo i valori SHAP
    explainer = shap.DeepExplainer(model, torch.tensor(matrix_train.todense(), dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(matrix_test.todense(), dtype=torch.float32),
                                        check_additivity=False)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.squeeze(shap_values)

    print(f"Dimensione di shap_values: {shap_values.shape}")

    # Converto i valori SHAP in un oggetto Explanation
    shap_values_explanation = shap.Explanation(values=shap_values,
                                               base_values=explainer.expected_value,
                                               data=matrix_test.todense(),
                                               feature_names=vocabulary_global)

    # Visualizzo i valori SHAP
    try:
        shap.plots.beeswarm(shap_values_explanation)
    except Exception as e:
        print(f"Errore durante la creazione del beeswarm plot: {e}")

    # Visualizzazione alternativa: bar plot
    try:
        # Bar plot per vedere l'importanza media delle feature
        shap.plots.bar(shap_values_explanation, max_display=100)
    except Exception as e:
        print(f"Errore durante la creazione del bar plot: {e}")



