import cv2
import numpy as np
import sqlite3

faceDetect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0);
rec = cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainingData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_PLAIN,5,1,0,4)

def getProfile(id):
    conn=sqlite3.connect("faceDB.db")
    cmd="SELECT * FROM People WHERE ID="+ str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        profile=getProfile(id)
        if(profile!=None):
            cv2.cv.PutText(cv2.cv.fromarray(img),"Name  : "+str(profile[1]),(x,y+h+30),font,(0,255,0));
            cv2.cv.PutText(cv2.cv.fromarray(img),"Age   : "+str(profile[2]),(x,y+h+60),font,(0,0,255));


    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()