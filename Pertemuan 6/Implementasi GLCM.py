from skimage.feature import graycomatrix, graycoprops
import cv2
import numpy as np

# Membaca gambar dalam grayscale
image = cv2.imread('wajah.jpg', 0)

# Pastikan gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan, pastikan path-nya benar.")
else:
    # Menghitung GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sudut-sudut yang digunakan
    glcm = graycomatrix(image, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)

    # Menghitung fitur tekstur dari GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    print(f'Contrast: {contrast}, Energy: {energy}')
