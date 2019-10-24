import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) < 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze) match_method (= bf ou flann)")
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
gray =  cv2.cvtColor(imgDistort,cv2.COLOR_BGR2GRAY)
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection et description des keypoints
pts, desc = kp.detectAndCompute(gray,None)
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")
# Calcul de l'appariement
t1 = cv2.getTickCount()
if detector == 1:
  #Distance de Hamming pour descripteur BRIEF (ORB)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
else:
  #Distance L2 pour descripteur M-SURF (KAZE)
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# Extraction de la liste des 2-plus-proches-voisins
matches = bf.knnMatch(desc1,desc2, k=2)
matchesDistorted = bf.knnMatch(desc1,desc, k=2)
# Application du ratio test
good = []
distances=[]
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])
    distances.append(m.distance)
print('Distance moyenne', sum(distances)/len(distances))
t2 = cv2.getTickCount()

goodDistorted = []
distancesD=[]
for m,n in matchesDistorted:
  if m.distance < 0.7*n.distance:
    goodDistorted.append([m])
    distancesD.append(m.distance)
print('Distance moyenne Distort',sum(distancesD)/len(distancesD))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Affichage des appariements qui respectent le ratio test
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
plt.imshow(img3),plt.title('%i appariements OK'%Nb_ok)
plt.show()

img4 = cv2.drawMatchesKnn(gray1,pts1,gray,pts,good,None,**draw_params)

Nb_ok = len(goodDistorted)
plt.imshow(img4),plt.title('%i appariements OK'%Nb_ok)
plt.show()


save_path = '~/home/julia/Downloads/ROB317/TP1_Features'

#name_of_file = input("What is the name of the file: ")
completeName = "RatioTest.txt"

#file1 = open(completeName, "a")
with open(completeName, "a") as file1:
  file1.write('Matches  ')
  #file1.close()

  #file1 = open(completeName, "w")
  #toFile=input(len(matchesDistorted))
  file1.write(''+sys.argv[1]+' ')
  file1.write('angle '+sys.argv[2]+' ')
  file1.write(str(len(goodDistorted))+' ')
  #toFile=input(len(matches))
  file1.write(str(len(good))+' ') 
  #toFile=input(len(matchesDistorted)/len(matches))
  file1.write(str(len(goodDistorted)/len(good))+' ')
  file1.write('Distance average '+str(sum(distances)/len(distances)) )
  file1.write('  Distance average dist '+str(sum(distancesD)/len(distancesD))+'\n' )
  file1.close()
  #file1 = open(completeName, "w+")
  #file1.write('\n'+str(len(matchesDistorted)/len(matches))+' ')
  #file1.close()

