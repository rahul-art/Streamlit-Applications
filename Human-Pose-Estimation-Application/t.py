import streamlit as st

from PIL import Image
import numpy as np
import cv2
#import mediapipe as mp
import os
import time
import posemodule as pm
import math
import random
DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


width = 368
height = 368
inWidth = width
inHeight = height

#net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
detector = pm.poseDetector()



st.title("Human Pose Estimation OpenCV")

st.text('Make Sure you have a clear image with all the parts clearly visible')

img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))
    
st.subheader('Original Image')
st.image(
    image, caption=f"Original Image", use_column_width=True
) 

thres = st.slider('Threshold for detecting the key points',min_value = 0,value = 20, max_value = 100,step = 5)

thres = thres/100

@st.cache
def poseDetector(frame):
    #rameWidth = frame.shape[1]
    #frameHeight = frame.shape[0]
    
    #net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), 
                                       #swapRB=True, crop=False))
    img = detector.findPose(frame)
    lmlist = detector.getPosition(frame,draw=False)
    #out = net.forward()
    #out = out[:, :19, :, :]
    
    #assert(len(BODY_PARTS) == out.shape[1])
    
    if len(lmlist)!=0:
        cv2.circle(img,(lmlist[25][1],lmlist[25][2]),8,(255,150,0),cv2.FILLED)
        cv2.circle(img,(lmlist[23][1],lmlist[23][2]),8,(255,150,0),cv2.FILLED)
         #print(lmlist[23])
        y1 = lmlist[25][2]
        y2 = lmlist[23][2]
        
        #length = a-b
        length = y2-y1
        if length>=-45 and f==0:
            f=1
        elif length<-50 and f==1:
            f=0
            count=count+1
            count70=count70-1
        elif length>=-57 and k==0:
            k=1
        elif length<-60 and k==1:
            k=0
            count70=count70+1
        
        print("Value of Y1  = {}".format(y1))
        print("Value of Y2  = {}".format(y2))
        print("Value of Length  = {}".format(length))
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,"100% " + "Total Number of Squats  "+str(int(count)),(50,60),cv2.FONT_HERSHEY_DUPLEX,0.5,
        (60,100,255),1)
        cv2.putText(img,"Calories Burnt  "+str(int(count)*0.32),(50,140),cv2.FONT_HERSHEY_DUPLEX,0.5,
        (60,100,255),1)
        #img = cv2.resize(img, (900,900))                    # Resize image
        
        
        xx = abs(length)
        progress = 0
        different = xx - 50
        print("Different value = {}".format(different))
        if different > 30:
            progress = 10
        elif different <= 25 and different > 20:
            progress = 30
        elif different <= 20 and different > 15:
            progress = 50
        elif different <= 15 and different > 10:
            progress = 60
        elif different <= 10 and different > 5:
            progress = 70
        elif different <= 5 and different > 2:
            progress = 90
        elif different <= 2 and different <=0:
            progress = 100
            
        img = prob_viz(progress , img, colors)
        
        print("xx value = {}".format(xx))
        print("-------------------------------progress value = {}".format(progress))
        
        #count70 = check(count, count70)
        cv2.putText(img,"70% " + "Total Number of Squats  "+str(int(count70)),(50,100),cv2.FONT_HERSHEY_DUPLEX,0.5,
        (60,100,255),1)
    return img


output = poseDetector(image)


st.subheader('Positions Estimated')
st.image(
       output, caption=f"Positions Estimated", use_column_width=True)
    
st.markdown('''
          # Author \n 
             Hey this is ** Pavan Kunchala ** I hope you like the application \n
            I am looking for ** Collabration ** or ** Freelancing ** in the field of ** Deep Learning ** and 
            ** Computer Vision ** \n
            If you're interested in collabrating you can mail me at ** pavankunchalapk@gmail.com ** \n
            You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/) \n
            You can check out my ** Github ** Profile from [here](https://github.com/Pavankunchala) \n
            You can also check my technicals blogs in ** Medium ** from [here](https://pavankunchalapk.medium.com/) \n
            If you are feeling generous you can buy me a cup of ** coffee ** from [here](https://www.buymeacoffee.com/pavankunchala)
             
            ''')
