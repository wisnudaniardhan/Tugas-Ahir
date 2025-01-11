import cv2

# memuat file haead cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# baca citra 
image = cv2.imread('wajah.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# deteksi wajah 
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# gambar bounding box disekitar wajah yang terdeteksi 
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# tampilkan hasil 
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()