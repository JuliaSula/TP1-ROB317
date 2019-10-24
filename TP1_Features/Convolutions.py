import numpy as np
import sys
import math
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe 
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1]  
    img2[y,x]= min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")
val = 0*img[y, x] - img[y, x-1] + img[y, x+1]
plt.subplot(121)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')




#Méthode filter2D 1
t1 = cv2.getTickCount()
kernel = np.array(([[0,-1,0], [-1, 5, -1], [0,-1,0]]))
img4 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")
plt.subplot(122)
plt.imshow(img4,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')
plt.show()

#Méthode directe derivée Partielle
t1 = cv2.getTickCount()
img3 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = 0*img[y, x] - img[y, x-1] + img[y, x+1]  
    img3[y,x]= min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")
plt.subplot(121)
plt.imshow(img3,cmap = 'gray')
plt.title('Convolution: Derivée partielle')

#Méthode filter2D  Derivée Partielle
t1 = cv2.getTickCount()
kernel = np.array(([[0,0,0], [-1, 0, 1], [0,0,0]]))
img5 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")
plt.subplot(122)
plt.imshow(img5,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('filter2D: Derivée partielle')
plt.show()


#Méthode directe intensite
t1 = cv2.getTickCount()
img6 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img11=cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img10=cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = 0*img[y, x] - img[y-1, x-1] + img[y-1, x+1]-2*img[y,x-1]+2*img[y, x+1]-img[y+1, x-1]+img[y+1, x+1]
    img10[y,x]= val
    val1 = 0*img[y, x]- img[y-1, x-1]-2*img[y-1, x]-img[y-1, x+1]+img[y+1, x-1]+2*img[y+1,x]+img[y+1, x+1]
    img11[y, x]=val1
    img6[y,x]= min(max(np.hypot(img10[y,x], img11[y,x]),0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")
plt.subplot(121)
plt.imshow(img6,cmap = 'gray')
plt.title('Méthode Directe: Gradient')

#Méthode filter2D  intensite
t1 = cv2.getTickCount()
kernel1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
img7 = cv2.filter2D(img,-1,kernel1)
kernel2 = np.array([[-1,-2,-1] ,[0,0,0], [1,2,1]])
img8 = cv2.filter2D(img,-1,kernel2)
img9= np.hypot(img7, img8)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")
plt.subplot(122)
plt.imshow(img9,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('filter2D: Gradient')
plt.show()
"""
img4=img3-img2
plt.imshow(img4,cmap = 'gray')
plt.title('Convolution - filter2D')
plt.show()
print(np.amax(img4))

img5=img2-img3
plt.imshow(img5,cmap = 'gray')
plt.title('Convolution - filter2D')
plt.show()
"""


