import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as plt





#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img10 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img11 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)


# Mettre ici le calcul de la fonction d'intérêt de Harris

sigma_def=1
alpha=0.06
#Calcul de la derivee par derivee de gaussian filter
img10=ndi.gaussian_filter(img,sigma=sigma_def,order=[0,1],output=np.float64, mode='nearest')
img11=ndi.gaussian_filter(img,sigma=sigma_def,order=[1,0],output=np.float64, mode='nearest')

#Plot des derivees
plt.subplot(121)
plt.imshow(img10,cmap = 'gray')
plt.title('Derivée Gaussian-x')

plt.subplot(122)
plt.imshow(img11,cmap = 'gray')
plt.title('Derivée Gaussian-y')
plt.show()
#a=0
#b=0
#c=0
#Moyenne par le kernel Gaussian
kernel=cv2.getGaussianKernel(2,sigma_def*2)
a=cv2.filter2D(img10**2, -1, kernel)
b=cv2.filter2D(img10*img11, -1, kernel)
c=cv2.filter2D(img11**2, -1, kernel)
"""for y in range(1,h-1):
  for x in range(1,w-1):
    for i in range (-1,1):
      for j in range (-1, 1):
        a+=img10[y+i, x+j]**2/9
        b+=img10[y+i, x+j]*img11[y+i, x+j]
        c+=img11[y+i, x+j]**2/9
  e=np.matrix([[a, b], [ b,c]])
    a=0
    b=0
    c=0"""
    #print(e)	
#Fonction d'interet: det- alpha*trace 
Theta=(a*c-b**2)-alpha*(a+b)**2


# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]

print("Numero de point d'interet:", np.size(Img_pts[Theta_ml_dil > 0]))
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()


imgt = cv2.imread('../Image_Pairs/Graffiti0.png')
gray = cv2.cvtColor(imgt,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst=cv2.cornerHarris(gray,2,3,0.06)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
imgt[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',imgt)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()



