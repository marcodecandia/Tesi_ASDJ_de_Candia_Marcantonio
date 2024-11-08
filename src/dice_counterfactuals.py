import dice_ml
import csv
import torch
import pickle
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from load_file import load_file_jsonl
import time
from collections import defaultdict

start_time = time.time()

# Carico il vocabolario globale
vocabulary_path = '../data/movies/vocabulary/vocabulary.txt'
with open(vocabulary_path, 'r') as f:
    vocabulary_global = [line.strip() for line in f.readlines()]

vocabulary_len = len(vocabulary_global)
print(f"Lunghezza del vocabolario: {vocabulary_len}")

# Inizializzo il modello
model = NeuralNetwork(vocabulary_len)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# Carico matrix_train e matrix_test da file pickle
with open("../data/movies/matrix_train.pkl", "rb") as f:
    matrix_train = pickle.load(f)

with open("../data/movies/matrix_test.pkl", "rb") as f:
    matrix_test = pickle.load(f)

print("matrix_train e matrix_test caricati da pickle.")


# Wrapper per il modello per adattarlo all'interfaccia di Dice
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if torch.is_tensor(x):
            x_tensor = x.clone().detach().float()
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            preds = self.get_output(x_tensor)
            return preds

    def get_output(self, input_instance):

        # Controllo del tipo di input
        if isinstance(input_instance, pd.DataFrame):
            input_tensor = torch.tensor(input_instance.to_numpy(), dtype=torch.float32)
        elif isinstance(input_instance, np.ndarray):
            input_tensor = torch.tensor(input_instance, dtype=torch.float32)
        elif isinstance(input_instance, torch.Tensor):
            input_tensor = input_instance.clone().detach().float()
        else:
            raise TypeError("input_instance deve essere un DataFrame, un array NumPy o un torch.Tensor")

        with torch.no_grad():
            out = self.model(input_tensor).clone().detach().float()
            return out.numpy()


# Definisco le feature continue e categoriali
continuous_features = vocabulary_global
categorical_features = []

# Carico le etichette (outcome)
train_labels = load_file_jsonl("../data/movies/train.jsonl")

# Assicuro che il numero di etichette corrisponda ai documenti
train_labels = train_labels[:matrix_train.shape[0]]

# Creo un DataFrame dei dati di training
data_df = pd.DataFrame(matrix_train.todense(), columns=continuous_features)

# Aggiungo la colonna target "_outcome_"
data_df["_outcome_"] = np.array(train_labels, dtype=np.float64)
print(f"Numero di colonne nel dataframe con l\'outcome: {len(data_df.columns)}")

# Verifico se tutte le feature sono presenti come colonne nel DataFrame
missing_features = set(continuous_features) - set(data_df.columns.tolist())
if missing_features:
    print(f"Le seguenti feature sono nel vocabolario ma non nel DataFrame: {missing_features}")
    # Filtro le feature mancanti
    continuous_features = [f for f in continuous_features if f in data_df.columns.tolist()]
    print(f"Le nuove feature continue aggiornate: {continuous_features}")
else:
    print("Tutte le feature sono correttamente presenti nel DataFrame.")

cinema_features = [
    "actor", "actress", "director", "screenplay", "script", "cinematography", "editing",
    "sound", "soundtrack", "lighting", "costume", "makeup", "production", "camera",
    "shot", "frame", "scene", "sequence", "visual", "cgi", "animation",
    "studio", "release", "premiere", "trailer", "crew", "cast",
    "dub", "subtitle", "dialogue", "role", "protagonist", "antagonist",
    "plot", "climax", "narrative", "theme", "symbolism", "adaptation", "remake",
    "sequel", "trilogy", "franchise", "series", "budget",
    "award", "festival", "oscar", "nomination", "critic", "rating", "score", "review",
    "audience", "classic", "masterpiece", "indie", "mainstream", "blockbuster",
    "opening", "closing", "pacing", "tone", "mood"
]

adjectives = [
    "amazing", "boring", "captivating", "stunning", "thrilling", "exciting", "dull",
    "tedious", "fantastic", "brilliant", "iconic", "legendary", "mediocre",
    "disappointing", "memorable", "intense", "emotional", "heartfelt", "powerful",
    "weak", "strong", "funny", "humorous", "dark", "gritty",
    "realistic", "surreal", "groundbreaking", "predictable", "unpredictable",
    "original", "fresh", "stale", "innovative", "charming",
    "beautiful", "ugly", "terrifying", "scary", "suspenseful", "uplifting",
    "depressing", "romantic", "intense", "witty", "slow",
    "fast", "complex", "simplistic", "raw", "refined", "polished", "harsh",
    "pleasant", "nostalgic"
]

genre_themes = [
    "horror", "thriller", "drama", "comedy", "action", "romance", "fantasy", "adventure", "mystery", "crime", "noir",
    "biography", "historical", "documentary", "war", "western", "musical", "animation", "anime",
    "family", "superhero", "survival", "disaster", "revenge", "justice",
    "redemption", "friendship", "love", "death", "life", "betrayal", "corruption",
    "freedom", "justice", "innocence", "conflict", "adventure", "journey",
    "exploration", "humanity", "nature", "technology", "civilization", "rebellion",
    "utopia", "mythology", "magic", "war", "hero", "villain", "antihero",
    "sacrifice", "bravery", "power", "religion", "philosophy", "existential",
    "moral", "society", "tradition", "culture"
]

actors_directors = [
    "spielberg", "tarantino", "hitchcock", "scorsese", "nolan", "kubrick",
    "coppola", "cameron", "ridley", "lucas", "zemeckis", "burton", "fincher",
    "bigelow", "coen", "lynch", "carpenter", "allen",
    "streep", "pacino", "depp", "dicaprio", "hathaway", "winslet", "kidman",
    "pitt", "clooney", "roberts", "lawrence", "damon", "williams", "mcconaughey",
    "johansson", "hanks", "swinton", "mirren", "watson",
    "watts", "gyllenhaal", "bale", "hardy", "redford", "harrison", "bullock",
    "dench"
]

descriptive_terms = [
    "masterpiece", "hit", "flop", "success", "failure", "thrilling", "suspenseful",
    "entertaining", "engrossing", "hilarious", "heartbreaking", "touching", "inspiring",
    "amusing", "shocking", "disturbing", "haunting", "majestic", "spectacular",
    "disappointing", "lifeless", "timeless", "forgettable", "overrated", "underrated",
    "immersive", "vivid", "surreal", "sharp", "edgy", "raw", "classic", "groundbreaking",
    "innovative", "stale", "refreshing", "bold", "ambitious", "modest", "engaging",
    "relatable", "flawed", "perfect", "cheesy", "melodramatic", "authentic", "visionary",
    "experimental", "straightforward", "complex", "layered",
    "cinematic", "aesthetic", "philosophical", "ambiguous", "symbolic",
    "metaphorical"
]

selected_features = cinema_features + adjectives + genre_themes + actors_directors + descriptive_terms

missing_features_1 = set(selected_features) - set(continuous_features)
if missing_features_1:
    print(f"Alcune feature sono nel selected ma non continuous")
    print(missing_features_1)
else:
    print("Tutte le feature sono correttamente presenti nel DataFrame.")

# Configurazione per DiCE
data = dice_ml.Data(dataframe=data_df,
                    continuous_features=continuous_features,
                    categorical_features=categorical_features,
                    outcome_name="_outcome_")

dice_model = dice_ml.Model(model=ModelWrapper(model), backend="PYT")

exp = dice_ml.Dice(data, dice_model, method="random")

# Dizionari per contare aumenti e diminuzioni in base all'outcome
feature_increase_count = defaultdict(int)
feature_decrease_count = defaultdict(int)
feature_changes_0_to_1 = defaultdict(lambda: {"increase": 0, "decrease": 0})
feature_changes_1_to_0 = defaultdict(lambda: {"increase": 0, "decrease": 0})

# Itero su tutte le righe di matrix_test (tutti i documenti)
for i in range(matrix_test.shape[0]):

    instance_array = np.squeeze((np.asarray(matrix_test[i].todense()))).reshape(1, -1)
    print(f"Generazione controfattuale per l'istanza {i}...")

    # Creo dataframe per istanza
    instance_df = pd.DataFrame(instance_array, columns=continuous_features)

    instance_tensor = torch.tensor(instance_array, dtype=torch.float32)
    prediction = ModelWrapper(model).predict(instance_tensor)

    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]

    outcome = int(prediction >= 0.5)
    print(f"Predizione per l'istanza {i}: {prediction}")

    # Genero controfattuali per l'istanza i-esima
    try:
        # Generazione dei controfattuali
        counterfactuals = exp.generate_counterfactuals(
            instance_df,
            total_CFs=1,
            desired_class="opposite",
            features_to_vary=selected_features,
            sparsity_weight=0.8
        )

        counterfactuals.visualize_as_dataframe()

        # Continua con il resto del codice per tracciare le modifiche
        if outcome == 0:
            change_dict = feature_changes_0_to_1
        elif outcome == 1:
            change_dict = feature_changes_1_to_0

        counterfactual_instance_array = counterfactuals.cf_examples_list[0].final_cfs_df.to_numpy()[0].flatten()
        changes = []

        for j, (orig_val, cf_val) in enumerate(zip(instance_array.ravel(), counterfactual_instance_array)):
            if orig_val != cf_val:
                feature_name = continuous_features[j]
                changes.append((feature_name, orig_val, cf_val))
                if cf_val > orig_val:
                    feature_increase_count[feature_name] += 1
                    change_dict[feature_name]["increase"] += 1
                else:
                    feature_decrease_count[feature_name] += 1
                    change_dict[feature_name]["decrease"] += 1

        # Stampa delle feature modificate
        print(f"Features modificate per l'istanza {i}:")
        for feature_name, orig_val, cf_val in changes:
            print(f"Feature: {feature_name}, Valore originale: {orig_val}, Nuovo valore: {cf_val}")

    except Exception as e:
        print(f"Eccezione per l'istanza {i}: {e}")
        print("Salto questa istanza e proseguo con la successiva")


print("Generazione dei controfattuali completata")

# Salvataggio dei risultati in CSV
with open("feature_changes.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Feature", "Increase count", "Decrease count", "Increase (0->1)", "Decrease (0->1)", "Increase (1->0)", "Decrease (1->0)"])
    for feature in continuous_features:
        if feature_increase_count[feature] > 0 or feature_decrease_count[feature] > 0:
            writer.writerow([
                feature,
                feature_increase_count[feature],
                feature_decrease_count[feature],
                feature_changes_0_to_1[feature]["increase"],
                feature_changes_0_to_1[feature]["decrease"],
                feature_changes_1_to_0[feature]["increase"],
                feature_changes_1_to_0[feature]["decrease"]
            ])

print("Conteggio modifiche salvato in feature_changes.csv")

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

    # Definisco modello per DiCE
    dice_model = dice_ml.Model(model=ModelWrapper(model), backend="PYT")

    # Inizializzo il generatore di controfattuali
    exp = dice_ml.Dice(data, dice_model, method="genetic")

    feature_modif_count = defaultdict(int)

    # Itero su tutte le righe di matrix_test (tutti i documenti)
    for i in range(matrix_test.shape[0]):

        instance_array = np.squeeze((np.asarray(matrix_test[i].todense()))).reshape(1, -1)
        print(f"Generazione controfattuale per l'istanza {i}...")

        # Creo dataframe per istanza
        instance = pd.DataFrame(instance_array, columns=continuous_features)

        instance_tensor = torch.tensor(instance_array, dtype=torch.float32)
        prediction = ModelWrapper(model).predict(instance_tensor)
        print(f"Predizione per l'istanza {i}: {prediction}")

        # Genero controfattuali per l'istanza i-esima
        counterfactuals = exp.generate_counterfactuals(instance, total_CFs=1, desired_class="opposite",
                                                       sparsity_weight=0.8)

        counterfactuals.visualize_as_dataframe()

        # Confronto i valori dell'istanza originale e del controfattuale
        counterfactual_instance_array = counterfactuals.cf_examples_list[0].final_cfs_df.to_numpy()[0].flatten()
        changes = []

        for j, (orig_val, cf_val) in enumerate(zip(instance_array.ravel(), counterfactual_instance_array)):
            if orig_val != cf_val:
                feature_name = continuous_features[j]
                changes.append((feature_name, orig_val, cf_val))

                # Conto quante volte ogni feature viene modificata
                feature_modif_count[feature_name] += 1

        # Stampo i risultati
        print(f"Features modificate per l'istanza {i}:")
        for feature_name, orig_val, cf_val in changes:
            print(f"Feature: {feature_name}, Valore originale: {orig_val}, Nuovo valore: {cf_val}")

    print("Generazione dei controfattuali completata")

    with open("feature_modif_count_1.csv", mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Feature", "Modification count"])
        for feature, count in feature_modif_count.items():
            writer.writerow([feature, count])

    print("Conteggio modifiche salvato in feature_modif_count_1.csv")
