import pandas as pd
features = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_features_85(1).csv")
labels = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_labels_85.csv")
# print(features)

import matplotlib.pyplot as plt

features['AGEP'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Nb de personnes')
plt.show()

# 2e Employee
features['COW'].value_counts().plot(kind='bar')
plt.title('Class of worker Distribution')
plt.xlabel('Worker type')
plt.ylabel('Nb de personnes')
plt.show()

# 3e  MAR
features['MAR'].value_counts().plot(kind='bar')
plt.title('Marie Distribution')
plt.xlabel('Marie type')
plt.ylabel('Nb de personnes')
plt.show()

data = pd.concat([features, labels], axis=1)
