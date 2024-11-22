import pandas as pd
features = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_features_85(1).csv")
labels = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_labels_85.csv")
# print(features)

import matplotlib.pyplot as plt

features['AGEP'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
