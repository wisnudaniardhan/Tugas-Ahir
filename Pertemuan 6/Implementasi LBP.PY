import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Membaca gambar dalam format grayscale
image = cv2.imread('wajah.jpg', 0)

# Menerapkan lokal binary pattern
radius = 1
n_points = 8 * radius  # Penetapan nilai n_points yang benar
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Menampilkan hasil
cv2.imshow('Lokal Binary Pattern', lbp.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
