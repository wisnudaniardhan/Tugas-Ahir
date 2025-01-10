import cv2

# Membaca gambar dalam grayscale
image = cv2.imread('wajah.jpg', 0)

#Deteksi tepi menggunakan canny
edges = cv2.Canny(image, 100, 200)

#Tampilkan hasil 
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()