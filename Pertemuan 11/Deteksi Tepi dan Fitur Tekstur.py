import cv2
import numpy as np

# Membaca gambar dalam graysicale
image = cv2.imread('wajah.jpg', cv2.IMREAD_GRAYSCALE)

# Menerapkan deteksi tepi menggunakan sobel
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) # sobel x
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) # sobel y

# Menampilkan hasil
cv2.imshow('sobel X', sobelx)
cv2.imshow('sobel Y', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows