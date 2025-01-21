import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path = 'wine/WineQT.csv'
wine_dataset = pd.read_csv(file_path)

# Periksa dataset
print(f"Kolom dalam dataset: {wine_dataset.columns}")
print(f"Sample dataset:\n{wine_dataset.head()}")

# 1. Menentukan kolom target dan fitur
target_column = 'quality'
if target_column not in wine_dataset.columns:
    raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset.")

# Pisahkan fitur (X) dan label (y)
X = wine_dataset.drop(columns=[target_column, 'Id']).values
y = wine_dataset[target_column].values

# Normalisasi fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Menyesuaikan nilai y agar dimulai dari 0
y = y - y.min()

# One-hot encode label (y)
num_classes = len(np.unique(y))
y_one_hot = to_categorical(y, num_classes=num_classes)

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# Membuat model Neural Network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),  # Dropout layer
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Akurasi Model pada dataset WineQT: {accuracy * 100:.2f}%')

 