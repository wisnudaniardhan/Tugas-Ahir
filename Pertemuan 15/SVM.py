import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Membaca dataset dari file CSV
file_path = r'c:\Users\bangbang\Downloads\wineQT.csv'

try:
    data = pd.read_csv(file_path)
    print("Dataset berhasil dibaca!")
except FileNotFoundError:
    print(f"File tidak ditemukan di lokasi: {file_path}")
    exit()

# 2. Menentukan kolom target (label) dan fitur
# Pada dataset wineQT, kolom target biasanya bernama 'quality'
target_column = 'quality'
if target_column not in data.columns:
    raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset.")

# Pisahkan fitur (X) dan target (y)
X = data.drop(columns=[target_column])  # Semua kolom kecuali kolom target
y = data[target_column]  # Kolom target

# 3. Encoding kolom kategori menjadi angka (jika perlu)
# Identifikasi kolom kategori dalam X
categorical_columns = X.select_dtypes(include=['object']).columns

# Konversi kolom kategori menjadi angka menggunakan LabelEncoder (jika ada kolom kategori)
label_encoder = LabelEncoder()
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Jika target (y) perlu diklasifikasi, pastikan untuk melakukan binning atau encoding jika datanya numerik
# Misalnya, jika kualitas wine berupa angka kontinu (regresi), dan ingin diubah menjadi klasifikasi:
# Contoh: Membagi kualitas anggur menjadi 'Low' (<=5), 'Medium' (6), 'High' (>=7)
y = y.apply(lambda val: 'Low' if val <= 5 else ('Medium' if val == 6 else 'High'))

# Encode target menjadi angka
y = label_encoder.fit_transform(y)

# 4. Membagi dataset menjadi data latih (training) dan data uji (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Melatih model SVM dengan kernel linear
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 6. Memprediksi data uji
y_pred = model.predict(X_test)

# 7. Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi SVM pada dataset WineQT: {accuracy * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
