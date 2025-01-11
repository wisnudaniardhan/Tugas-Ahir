import cv2

# Membaca gambar dalam graysicale
image = cv2.imread('wajah.jpg', cv2.IMREAD_GRAYSCALE)

# Menerapkan deteksi tepi canny
edges = cv2.Canny(image, 100, 200)

# Menampilkan hasil
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows