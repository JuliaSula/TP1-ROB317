import numpy as np
import os.path
import sys
import math
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from matplotlib import pyplot as plt
import cv2

if len(sys.argv) < 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)

#Lecture de la paire d'images
img = cv2.imread('../Image_Pairs/torb_small1.png')
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
img2 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)

#Distortion of the image
rows,cols,ch = img.shape
pt= (cols/2, rows/2)
theta=float(sys.argv[2])
r=cv2.getRotationMatrix2D(pt, theta, 1.0 )
imgDistort=cv2.warpAffine(img, r,(cols, rows))

#Plot image deforme
plt.imshow(imgDistort, cmap = 'gray')
plt.show()

#Début du calcul
t1 = cv2.getTickCount()

#Création des objets "keypoints"
if detector == 1:
  kp = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                       scaleFactor = 1.2,#Par défaut : 1.2
                       nlevels = 8)#Par défaut : 8
  kp1 = cv2.ORB_create(nfeatures=500,
                        scaleFactor = 1.2,
                        nlevels = 8)
  print("Détecteur : ORB")
else:
  kp = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")

#Conversion en niveau de gris
gray =  imgDistort
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#Détection et description des keypoints
pts, desc = kp.detectAndCompute(gray,None)
pts1, desc1 = kp1.detectAndCompute(gray1,None)

#Les points non appariés apparaîtront en gris 
img = cv2.drawKeypoints(gray, pts, None, color=(127,127,127), flags=0)
img1 = cv2.drawKeypoints(gray1, pts1, None, color=(127,127,127), flags=0)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")

# Calcul de l'appariement
t1 = cv2.getTickCount()
if detector == 1:
  #Distance de Hamming pour descripteur BRIEF (ORB)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
else:
  #Distance L2 pour descripteur M-SURF (KAZE)
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matchesDistorted = bf.match(desc,desc1)

# Tri des appariemements 
matchesDistorted = sorted(matchesDistorted, key = lambda x:x.distance)
distancesD=[]

#Creation de Vecteur de Distances
for m in matchesDistorted:
    distancesD.append(m.distance)


t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Trace les N meilleurs appariements
#img4 = cv2.drawMatches(img1,pts1,img,pts,matchesDistorted,None,flags=2)
#plt.imshow(img4),plt.title('%i meilleurs appariements'%len(matchesDistorted))
#plt.show()



distX=[]
distY=[]
dist=[]

bon=0
for m in matchesDistorted:
	#Index des Descripteurs apparies
	Idx=m.queryIdx
	Idx1=m.trainIdx
	#Recuperation des cordonnes a rotationer
	(x,y)=pts1[Idx1].pt
	#Rotation
	ox, oy = pt																#Centre de l'image
	qx = ox + math.cos(theta*math.pi/180) * (x - ox) + math.sin(theta*math.pi/180) * (y - oy)
	qy = oy - math.sin(theta*math.pi/180) * (x - ox) + math.cos(theta*math.pi/180) * (y - oy)
	#Calcul de distances en chaque axis
	distX=qx-pts[Idx].pt[0]
	distY=qy-pts[Idx].pt[1]
	#Calcul de la distance absolute
	dist.append(math.sqrt(distX**2+distY**2))
for i in dist:
	if(i<1):
		bon=bon+1
print(bon/len(dist))
print(min(dist))
