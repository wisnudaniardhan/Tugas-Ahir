import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'wine/WineQT.csv'  # Ganti dengan lokasi file dataset Anda
cars_dataset = pd.read_csv(file_path)

# Periksa nama kolom dalam dataset
print(f"Kolom dalam dataset: {cars_dataset.columns}")

# Pastikan kolom target yang benar digunakan
target_column = 'quality'
if target_column not in cars_dataset.columns:
    raise KeyError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

# Hapus kolom 'Id' karena tidak relevan
data = cars_dataset.drop(columns=['Id'])

# Split features (X) dan target (y)
X = data.drop(columns=[target_column]).values  # Fitur
y = data[target_column].values  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize SVM model with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy * 100:.2f}%')
