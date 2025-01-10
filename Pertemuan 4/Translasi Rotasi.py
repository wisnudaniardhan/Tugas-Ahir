import cv2 
import numpy as np

# Membaca Gambar
image = cv2.imread('wajah.jpg')

# Mendapatkan dimensi gambar 
(h, w) = image.shape[:2]

# Teranslasi
M_translation = np.float32([[1, 0, 50], [0, 1, 100]])
traslated_image = cv2.warpAffine(image, M_translation, (w, h))

# Rotasi 45 Derajat 
center = (w // 2, h // 2)
M_rotation = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, M_rotation, (w, h))

# Menampilkan hasil 
cv2.imshow('Translated Image', traslated_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()