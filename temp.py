import cv2
import sys
import matplotlib.pyplot as plt
img = plt.imread("MM.jpg")  #burada secilen resmi yukledik

plt.imshow(img) #resmin yuklendigi kontrolu yapıldı
yuz_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #hazir paket ile yuzu tanimladik
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #resmi gri renge cevirdik ki bilgisayar okuyabilsin
yuz = yuz_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3) #belirtilen alanlarda yuz taraması yapıldı
for (x,y,w,h) in yuz: 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255 ), 2)
    roi_gray = gray[y:y+h, x:x+w] 
    roi_color = img[y:y+h, x:x+w] 
print("{0} Yüz Bulundu!".format(len(yuz)))   
