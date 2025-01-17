import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Path ke file dataset
file_path = r'c:\Users\bangbang\Downloads\lung_cancer_data.csv'

# Validasi apakah file ada
if not os.path.exists(file_path):
    print(f"File tidak ditemukan di lokasi: {file_path}")
    exit()

# Membaca dataset
try:
    diamond = pd.read_csv(file_path)
except Exception as e:
    print(f"Error saat membaca file CSV: {e}")
    exit()

# Periksa kolom yang tersedia
print("Kolom dalam dataset:", diamond.columns)

# Tentukan kolom target (label)
target_column = 'Stage_of_Cancer'  # Ganti sesuai dengan kolom target Anda

# Cek apakah dataset memiliki kolom target
if target_column not in diamond.columns:
    print(f"Kolom target '{target_column}' tidak ditemukan dalam dataset.")
    exit()

# Konversi label kategori menjadi angka
try:
    label_encoder = LabelEncoder()
    diamond[target_column] = label_encoder.fit_transform(diamond[target_column])
except Exception as e:
    print(f"Error dalam konversi label: {e}")
    exit()

# Identifikasi kolom kategori (selain kolom target)
categorical_columns = diamond.select_dtypes(include=['object']).columns

# Konversi semua kolom kategori menjadi angka menggunakan LabelEncoder
for col in categorical_columns:
    print(f"Encoding kolom kategori: {col}")
    diamond[col] = label_encoder.fit_transform(diamond[col])

# Ambil fitur (X) dan label (y)
X = diamond.drop(columns=[target_column]).values  # Semua kolom kecuali kolom target
y = diamond[target_column].values                 # Kolom target sebagai label

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Latih model pada data training
knn.fit(X_train, y_train)

# Prediksi pada data testing
y_pred = knn.predict(X_test)

# Hitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model KNN pada dataset: {accuracy * 100:.2f}%")

