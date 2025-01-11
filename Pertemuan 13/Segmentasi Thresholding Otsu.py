import cv2
import numpy as np

# Membaca gambar dalam grayscale
image = cv2.imread('wajah.jpg', 0)

# Menerapkan otsu's thresholding
ret, outsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

# Menampilkan hasil
cv2.imshow('Otsu Thresholded', outsu_image)
cv2.waitKey(0)
cv2.destroyAllWindows()