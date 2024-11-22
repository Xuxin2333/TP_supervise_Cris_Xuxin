import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Read data
features = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_features_85(1).csv")
labels = pd.read_csv("./JeuDeDonnees/alt_acsincome_ca_labels_85.csv")

# Print content
print("DATABASE 1")
print(features.head())
print(features.columns)

# Delete missing values - NULL, If 'MAR' or 'COW' contain NULL values 
features_df = features.dropna(subset=['MAR', 'COW'])

# Create relation between 'MAR' and 'COW' 
features_relation = features_df.groupby(['MAR', 'COW']).size().reset_index(name='QUANTITÉ')
print(features_relation)

# Create Graphic
plt.figure(figsize=(10, 6))
sns.barplot(data=features_relation, x='MAR', y='QUANTITÉ', hue='COW', palette='viridis')
plt.title('Relation entre MAR (État Civil) et COW (Occupation)')
plt.xlabel('MAR (État Civil)')
plt.ylabel('QUANTITÉ PERSONNES')
plt.legend(title='COW (Occupation)')
plt.show()

print("")
print("DATABASE 2")
# print(labels.head())  # Show first lines from the dataset
print(labels.columns)   # Show all the columns on the dataset



print("\n-----------------------------")
print("  NO SHUFFLE AND DIVIDED DATA  ")
print("-----------------------------\n")
# Divide data
X_train_NoShuffle, X_test_NoShuffle, y_train_NoShuffle, y_test_NoShuffle = train_test_split(
    features,           # X_train - X_test Características (features)
    labels,             # y_train - y_test  (labels)
    test_size=0.2,      # Size test set (20%)
    random_state=42,    # Para reproducibilidad
    shuffle=False       # Mix data FALSE
)

# Print Size SET and Result
print("\nSize TRAINING set:", X_train_NoShuffle.shape)
print("TRAINING set:")
print(X_train_NoShuffle) # Print Results

print("\nSize TEST set:", X_test_NoShuffle.shape, "\n")
print("TEST set:")
print(X_test_NoShuffle) # Print Results



print("\n-----------------------------")
print("    SHUFFLE AND DIVIDED DATA   ")
print("-----------------------------\n")
# Shuffle and Divide data
X_train, X_test, y_train, y_test = train_test_split(
    features,           # X_train - X_test Características (features)
    labels,             # y_train - y_test  (labels)
    test_size=0.2,      # Size test set (20%)
    random_state=42,    # Para reproducibilidad
    shuffle=False       # Mix data TRUE
)

print("\nSize TRAINING set:", X_train.shape)
print("TRAINING set:")
print(X_train) # Print Results

print("\nSize TEST set:", X_test.shape, "\n")
print("TEST set:")
print(X_test) # Print Results



# from sklearn.utils import shuffle
# datos_shuffled = shuffle(features)
# To standaliser the datas
my_scaler = StandardScaler()
X_train_Standed = my_scaler.fit_transform(X_train.select_dtypes(include=['float64','int64']))
X_test_Standed = my_scaler.fit_transform(X_test.select_dtypes(include=['float64','int64']))
joblib.dump (my_scaler, 'my_scaler.joblib')
