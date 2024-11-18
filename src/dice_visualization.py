import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carico il file csv relativo ai conteggi delle modifiche delle features
feature_modif = pd.read_csv("feature_changes.csv")

# Grafico a barre impilato

plt.figure(figsize=(12, 6))
feature_modif["Total changes"] = feature_modif[["Increase count", "Decrease count"]].sum(axis=1)

colors = {
    'Increase (0->1)': '#006400',  # Verde scuro
    'Decrease (1->0)': '#FF7F7F',  # Verde chiaro
    'Increase (1->0)': '#8B0000',  # Rosso scuro
    'Decrease (0->1)': '#90EE90'   # Rosso chiaro
}

# Seleziono le 20 feature con più modifiche
top_features = feature_modif.nlargest(20, "Total changes")

# Realizzo grafico a barre verticale impilato
top_features.plot(
    x="Feature",
    y=['Increase (0->1)', 'Decrease (0->1)', 'Increase (1->0)', 'Decrease (1->0)'],
    kind="bar",
    stacked=True,
    color=[colors['Increase (0->1)'], colors['Decrease (1->0)'], colors['Increase (1->0)'], colors['Decrease (0->1)']]
)
plt.title("Top 20 Feature più modificate per cambiare l'outcome")
plt.xlabel("Feature")
plt.ylabel("Conteggio delle modifiche")
plt.legend(title="Tipo di modifica")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calcolo un punteggio normalizzato per le feature

# Calcolo un punteggio positivo e negativo per ciascuna feature
feature_modif["Positive Score"] = feature_modif["Increase (0->1)"] + feature_modif["Decrease (1->0)"]
feature_modif["Negative Score"] = feature_modif["Increase (1->0)"] + feature_modif["Decrease (0->1)"]

# Punteggio totale
feature_modif["Score"] = feature_modif["Positive Score"] - feature_modif["Negative Score"]
max_score = feature_modif["Score"].max()
min_score = feature_modif["Score"].min()

# Normalizzo tra massimo e minimo
feature_modif["Word Rate"] = (feature_modif["Score"] - min_score) / (max_score - min_score)

# Ordino le feature per il punteggio calcolato
feature_modif_sorted = feature_modif.sort_values(by="Score", ascending=False)

# Seleziono le 15 feature più positive e le 15 più negative
top_positive_features = feature_modif_sorted.head(15)
top_negative_features = feature_modif_sorted.tail(15)

top_features_combined = pd.concat([top_positive_features, top_negative_features])

# Grafico a barre orizzontale

# Configuro la colormap da verde a rosso
colormap = sns.diverging_palette(10, 150, as_cmap=True)
norm = plt.Normalize(0, 1)
colors = colormap(norm(top_features_combined["Word Rate"]))

# Realizzo grafico a barre orizzontale con gradiente di colore
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top_features_combined["Feature"], top_features_combined["Word Rate"], color=colors)
ax.invert_yaxis()

ax.set_title("Top 15 Feature più positive e negative")
ax.set_xlabel("Punteggio normalizzato (Word Rate)")
ax.set_ylabel("Feature")

# Aggiungi la colorbar alla figura
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Word Rate")

plt.tight_layout()
plt.show()
