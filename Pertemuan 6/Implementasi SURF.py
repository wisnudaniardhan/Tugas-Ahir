import cv2

# Membaca gambar
image = cv2.imread('wajah.jpg')

# Inisialisasi objek ORB (alternatif dari SURF)
orb = cv2.ORB_create()

# Mendeteksi keypoints dan komputasi deskriptor
keypoints, descriptors = orb.detectAndCompute(image, None)

# Menggambarkan keypoints di citra
orb_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Menampilkan hasil
cv2.imshow('ORB Features', orb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
