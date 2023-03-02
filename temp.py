import cv2
import sys
import matplotlib.pyplot as plt
img = plt.imread(r"C:\Users\iremo\OneDrive\Desktop\lessons\ai\MM.jpg")  

plt.imshow(img)
yuz_cascade = cv2.CascadeClassifier(r"C:\Users\iremo\OneDrive\Desktop\lessons\ai\data\haarcascades\haarcascade_frontalface_alt.xml")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
yuz = yuz_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
for (x,y,w,h) in yuz: 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255 ), 2)
    roi_gray = gray[y:y+h, x:x+w] 
    roi_color = img[y:y+h, x:x+w] 
print("[INFO] {0} Yüz Bulundu!".format(len(yuz)))   