import numpy as np
import os.path
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from matplotlib import pyplot as plt
import cv2
import sys
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
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
pt= (cols/2, rows/2)
r=cv2.getRotationMatrix2D(pt, float(sys.argv[2]), 1.0 )
imgDistort=cv2.warpAffine(img, r,(cols,rows))
plt.imshow(imgDistort, cmap = 'gray')
print(ch)
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
  kp2 = cv2.ORB_create(nfeatures=500,
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
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray =  imgDistort
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection et description des keypoints
pts, desc = kp.detectAndCompute(gray,None)
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
#Les points non appariés apparaîtront en gris 
img = cv2.drawKeypoints(gray, pts, None, color=(127,127,127), flags=0)
img1 = cv2.drawKeypoints(gray1, pts1, None, color=(127,127,127), flags=0)
img2 = cv2.drawKeypoints(gray2, pts2, None, color=(127,127,127), flags=0)
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
matches = bf.match(desc1,desc2)
matchesDistorted = bf.match(desc1,desc)


# Tri des appariemements 
matchesDistorted = sorted(matchesDistorted, key = lambda x:x.distance)
matches = sorted(matches, key = lambda x:x.distance)
distancesD=[]
distances=[]
for m in matchesDistorted:
    distancesD.append(m.distance)
for m in matches:
	distances.append(m.distance)
print('Distance moyenne',sum(distances)/len(distances))
print('Distance moyenne Distort',sum(distancesD)/len(distancesD))

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Trace les N meilleurs appariements
img3 = cv2.drawMatches(img1,pts1,img2,pts2,matches,None,flags=2)
plt.imshow(img3),plt.title('%i meilleurs appariements'%len(matches))
plt.show()

img4 = cv2.drawMatches(img1,pts1,img,pts,matchesDistorted,None,flags=2)
plt.imshow(img4),plt.title('%i meilleurs appariements'%len(matchesDistorted))
plt.show()

print(len(matchesDistorted))
print(len(matches))




save_path = '~/home/julia/Downloads/ROB317/TP1_Features'

#name_of_file = input("What is the name of the file: ")
completeName = "text.txt"

#file1 = open(completeName, "a")
with open(completeName, "a") as file1:
  file1.write('Matches  ')
  #file1.close()

  #file1 = open(completeName, "w")
  #toFile=input(len(matchesDistorted))
  file1.write(''+sys.argv[1]+' ')
  file1.write('angle '+sys.argv[2]+' ')
  file1.write(str(len(matchesDistorted))+' ')
  #toFile=input(len(matches))
  file1.write(str(len(matches))+' ') 
  #toFile=input(len(matchesDistorted)/len(matches))
  file1.write(str(len(matchesDistorted)/len(matches))+' ')
  file1.write('Distance average '+str(sum(distances)/len(distances)) )
  file1.write('  Distance average dist '+str(sum(distancesD)/len(distancesD))+'\n' )
  file1.close()
  #file1 = open(completeName, "w+")
  #file1.write('\n'+str(len(matchesDistorted)/len(matches))+' ')
  #file1.close()
