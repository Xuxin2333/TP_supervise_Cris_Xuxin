import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
features = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_features_85(1).csv")
labels = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_labels_85.csv")

# Verificar las primeras filas y las columnas
print("DATABASE 1")
print(features.head())
print(features.columns)

features['AGEP'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Nb de personnes')
plt.show()
# Limpiar los datos eliminando filas con valores nulos en 'MAR' y 'COW'
features_df = features.dropna(subset=['MAR', 'COW'])

# Agrupar los datos por 'MAR' (estado civil) y 'COW' (ocupación)
features_relation = features_df.groupby(['MAR', 'COW']).size().reset_index(name='QUANTITÉ')
print(features_relation)

# Crear el gráfico
plt.figure(figsize=(10, 6))
sns.barplot(data=features_relation, x='MAR', y='QUANTITÉ', hue='COW', palette='viridis')
plt.title('Relation entre MAR (État Civil) et COW (Occupation)')
plt.xlabel('MAR (État Civil)')
plt.ylabel('QUANTITÉ PERSONNES')
plt.legend(title='COW (Occupation)')
plt.show()


print("")
print("DATABASE 2")
# print(labels.head())  # Muestra las primeras filas del dataset
print(labels.columns)  # Lista todas las columnas disponibles


# from sklearn.utils import shuffle
# datos_shuffled = shuffle(features)

data = pd.concat([features, labels], axis=1)

