import cv2
import numpy as np

# Membaca gambar dalam grayscale
image = cv2.imread('wajah.jpg', cv2.IMREAD_GRAYSCALE)

# Mengonversi gambar grayscale ke BGR
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Menerapkan Harris Corner Detection
gray = np.float32(image)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Thresholding untuk menandai sudut
image_color[corners > 0.01 * corners.max()] = [0, 0, 255]

# Menampilkan hasil
cv2.imshow('Harris Corners', image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
