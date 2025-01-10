import cv2
import numpy as np

# Membaca Gmabar 
image = cv2.imread('wajah.jpg')

# Mendefinisikan empat titik sudut cerita asli 
points1 = np.float32([[5, 65], [368, 52], [28, 387], [389, 390]])

# Mendefinisikan sudut citra baru 
points2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# Mendapatkan matariks transformasi perspektif
M_perspective = cv2.getPerspectiveTransform(points1, points2)

# Melakukan trasformasi perspektif
perspective_transformed_image = cv2.warpPerspective(image, M_perspective, (300, 300))

# Menampilkan hasil
cv2.imshow('Perspective Transformed Image', perspective_transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()