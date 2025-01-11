import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('wajah.jpg')
Z = image.reshape((-1, 3))

# Konversi ke float32
Z = np.float32(Z)

# Kriteria k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Konversi Kembali ke unit8 dan reshape
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((image.shape))

# menampilkan hasil 
cv2.imshow('Watershed Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()