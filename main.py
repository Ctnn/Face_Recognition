import cv2
import numpy as np
import face_recognition

imgdwayne=face_recognition.load_image_file("pic/dwayne.jpg")
imgdwayne=cv2.cvtColor(imgdwayne,cv2.COLOR_BGR2RGB)

imgtest= face_recognition.load_image_file("pic/testdwayne.jpg")
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceLocation= face_recognition.face_locations(imgdwayne)[0]
encodewayne=face_recognition.face_encodings(imgdwayne)[0]
cv2.rectangle(imgdwayne,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(255,0,255),2)

faceLocationTest= face_recognition.face_locations(imgtest)[0]
encodewayneTest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(255,0,255),2)




cv2.imshow("Dwayne Johnson",imgdwayne)
cv2.imshow("Dwayne Test",imgtest)
cv2.waitKey(0)