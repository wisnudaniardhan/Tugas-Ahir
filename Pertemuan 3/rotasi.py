import cv2

# Membaca gambar
image = cv2.imread('wajah.jpg')

# Mendapatkan dimensi gambar 
(h, w) = image.shape[:2]

# Menentukan pusat gambar 
center = (w // 2, h // 2)

# Menentukan matriks rotasi
M = cv2.getRotationMatrix2D(center, 45, 1.0)

# Melakukan rotasi 
rotated_image = cv2.warpAffine(image, M, (w, h))

# Menampilkan Hasil 
cv2.imshow('Rotated Imae', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()