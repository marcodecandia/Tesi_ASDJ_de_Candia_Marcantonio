import torch
import numpy as np
import shap
import pickle
import pandas as pd
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns
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

shap_values_array = np.array(shap_values_explanation.values)
data_array = np.array(shap_values_explanation.data)
feature_names_array = np.array(shap_values_explanation.feature_names)

# calcolo i valori medi dei SHAP
mean_shap_values = np.mean(shap_values_array, axis=0)

# estraggo i 10 valori SHAP più alti e più bassi
mean_shap_values_array_sorted = np.argsort(mean_shap_values)
top_indices = mean_shap_values_array_sorted[-10:]
low_indices = mean_shap_values_array_sorted[:10]

# Filtro i nomi delle feature per i 10 valori più alti e più bassi
top_shap_values = shap_values_array[:, top_indices]
top_data = data_array[:, top_indices]
top_feature_names = feature_names_array[top_indices]

low_shap_values = shap_values_array[:, low_indices]
low_data = data_array[:, low_indices]
low_feature_names = feature_names_array[low_indices]

# Creo due oggetti Explanation, uno riferito ai valori più alti e uno ai valori più bassi
top_shap_values_explanation = shap.Explanation(values=top_shap_values,
                                               base_values=explainer.expected_value,
                                               data=top_data,
                                               feature_names=top_feature_names)

low_shap_values_explanation = shap.Explanation(values=low_shap_values,
                                               base_values=explainer.expected_value,
                                               data=low_data,
                                               feature_names=low_feature_names)
# Visualizzo i SHAP values per le 10 feature più positive e negative

# bar plot
plt.figure(figsize=(10, 6))

plt.barh(top_feature_names, top_shap_values.mean(axis=0), color='blue')
plt.xlabel("SHAP value medio")
plt.title("SHAP value medio delle 10 feature più positive")
plt.show()

plt.barh(low_feature_names, low_shap_values.mean(axis=0), color='red')
plt.xlabel("SHAP value medio")
plt.title("SHAP value medio delle 10 feature più negative")
plt.show()

# heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(top_shap_values, cmap='coolwarm', xticklabels=top_feature_names, yticklabels=["SHAP values più positivi"])
plt.title("Heatmap dei SHAP value delle 10 feature più positive")
plt.xlabel("Feature")
plt.ylabel("SHAP value")
plt.show()

sns.heatmap(low_shap_values, cmap='coolwarm', xticklabels=low_feature_names, yticklabels=["SHAP values più negativi"])
plt.title("Heatmap dei SHAP value delle 10 feature più negative")
plt.xlabel("Feature")
plt.ylabel("SHAP value")
plt.show()

# beeswarm plot
plt.figure(figsize=(10, 6))
df_top = pd.DataFrame(top_shap_values,
                      columns=top_feature_names)
df_top_melted = df_top.melt(var_name='Feature', value_name='SHAP value')
sns.swarmplot(x='SHAP value', y='Feature', data=df_top_melted, size=4)
plt.title("Beeswarm plot dei valori più alti")
plt.xlabel("SHAP value")
plt.ylabel("Features")
plt.show()

df_low = pd.DataFrame(low_shap_values,
                      columns=low_feature_names)
df_low_melted = df_low.melt(var_name='Feature', value_name='SHAP value')
sns.swarmplot(x='SHAP value', y='Feature', data=df_low_melted, size=4)
plt.title("Beeswarm plot dei valori più bassi")
plt.xlabel("SHAP value")
plt.ylabel("Features")
plt.show()


# beeswarm plot
#try:
#    shap.plots.beeswarm(top_shap_values_explanation)
#    shap.plots.beeswarm(low_shap_values_explanation)
#except Exception as e:
#    print(f"Errore durante la creazione del beeswarm plot: {e}")

# bar plot
#try:
#    # Bar plot per vedere l'importanza media delle feature
#    shap.plots.bar(top_shap_values_explanation)
#    shap.plots.bar(low_shap_values_explanation)
#except Exception as e:
#    print(f"Errore durante la creazione del bar plot: {e}")

# heatmap plot
#try:
#    shap.plots.heatmap(top_shap_values_explanation)
#    shap.plots.heatmap(low_shap_values_explanation)
#except Exception as e:
#    print(f"Errore durante la creazione dell'heatmap plot: {e}")


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

    shap_values_array = np.array(shap_values_explanation.values)
    data_array = np.array(shap_values_explanation.data)
    feature_names_array = np.array(shap_values_explanation.feature_names)

    # calcolo i valori medi dei SHAP
    mean_shap_values = np.mean(shap_values_array, axis=0)

    # estraggo i 10 valori SHAP più alti e più bassi
    mean_shap_values_array_sorted = np.argsort(mean_shap_values)
    top_indices = mean_shap_values_array_sorted[-10:]
    low_indices = mean_shap_values_array_sorted[:10]

    # Filtro i nomi delle feature per i 10 valori più alti e più bassi
    top_shap_values = shap_values_array[:, top_indices]
    top_data = data_array[:, top_indices]
    top_feature_names = feature_names_array[top_indices]

    low_shap_values = shap_values_array[:, low_indices]
    low_data = data_array[:, low_indices]
    low_feature_names = feature_names_array[low_indices]

    # Creo due oggetti Explanation, uno riferito ai valori più alti e uno ai valori più bassi
    top_shap_values_explanation = shap.Explanation(values=top_shap_values,
                                                   base_values=explainer.expected_value,
                                                   data=top_data,
                                                   feature_names=top_feature_names)

    low_shap_values_explanation = shap.Explanation(values=low_shap_values,
                                                   base_values=explainer.expected_value,
                                                   data=low_data,
                                                   feature_names=low_feature_names)
    # Visualizzo i SHAP values per le 10 feature più positive e negative

    # bar plot
    plt.figure(figsize=(10, 6))

    plt.barh(top_feature_names, top_shap_values.mean(axis=0), color='blue')
    plt.xlabel("SHAP value medio")
    plt.title("SHAP value medio delle 10 feature più positive")
    plt.show()

    plt.barh(low_feature_names, low_shap_values.mean(axis=0), color='red')
    plt.xlabel("SHAP value medio")
    plt.title("SHAP value medio delle 10 feature più negative")
    plt.show()

    # heatmap plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_shap_values, cmap='coolwarm', xticklabels=top_feature_names,
                yticklabels=["SHAP values più positivi"])
    plt.title("Heatmap dei SHAP value delle 10 feature più positive")
    plt.xlabel("Feature")
    plt.ylabel("SHAP value")
    plt.show()

    sns.heatmap(low_shap_values, cmap='coolwarm', xticklabels=low_feature_names,
                yticklabels=["SHAP values più negativi"])
    plt.title("Heatmap dei SHAP value delle 10 feature più negative")
    plt.xlabel("Feature")
    plt.ylabel("SHAP value")
    plt.show()

    # beeswarm plot
    plt.figure(figsize=(10, 6))
    df_top = pd.DataFrame(top_shap_values,
                          columns=top_feature_names)
    df_top_melted = df_top.melt(var_name='Feature', value_name='SHAP value')
    sns.swarmplot(x='SHAP value', y='Feature', data=df_top_melted, size=4)
    plt.title("Beeswarm plot dei valori più alti")
    plt.xlabel("SHAP value")
    plt.ylabel("Features")
    plt.show()

    df_low = pd.DataFrame(low_shap_values,
                          columns=low_feature_names)
    df_low_melted = df_low.melt(var_name='Feature', value_name='SHAP value')
    sns.swarmplot(x='SHAP value', y='Feature', data=df_low_melted, size=4)
    plt.title("Beeswarm plot dei valori più bassi")
    plt.xlabel("SHAP value")
    plt.ylabel("Features")
    plt.show()
