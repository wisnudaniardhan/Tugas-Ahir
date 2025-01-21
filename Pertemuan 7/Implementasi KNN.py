import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path = 'wine/WineQT.csv'  # Ganti dengan lokasi file dataset Anda
cars_dataset = pd.read_csv(file_path)

# Periksa nama kolom dalam dataset
print(f"Kolom dalam dataset: {cars_dataset.columns}")

# Pastikan kolom target yang benar digunakan
target_column = 'quality'  # Ubah ke nama kolom target yang benar
if target_column not in cars_dataset.columns:
    raise KeyError(f"Kolom target '{target_column}' tidak ditemukan di dataset.")

# Encode categorical features and labels
label_encoders = {}
encoded_data = cars_dataset.copy()
for column in encoded_data.columns:
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(encoded_data[column])
    label_encoders[column] = le

# Extract features and labels
X = encoded_data.drop(columns=[target_column, 'Id']).values  # Hapus kolom target dan kolom 'Id'
y = encoded_data[target_column].values

# Normalize features
X = X / np.max(X)

# One-hot encode labels
num_classes = len(np.unique(y))
y_one_hot = to_categorical(y, num_classes=num_classes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# Define a simple Fully Connected Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer matches number of features
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Akurasi Model: {accuracy * 100:.2f}%')





