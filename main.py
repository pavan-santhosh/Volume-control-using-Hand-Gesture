'''import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self,mode=False,maxHands = 2, detectionCon=0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        #self.hands = self.mpHands.Hands(self.mode , self.maxHands,self.detectionCon, self.trackCon)
        self.complexity = 1
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]


            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ =="__main__":
    main()

'''#working handtracking




'''

import cv2
import time
import numpy as np
#import HandTrackingModule as htm
import HandTracking as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]




wCam, hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)
vol = 0
volBar = 400
volPer = 0
while True:
  success, img = cap.read()
  img = detector.findHands(img)
  lmlist = detector.findPosition(img)
  #print(lmlist)
  if len(lmlist)!=0:
    x1, y1 = lmlist[4][1], lmlist[4][2]
    x2, y2 = lmlist[8][1], lmlist[8][2]

    cx, cy = (x1+x2)//2, (y1+y2)//2
    
    cv2.circle(img, (x1,y1), 15, (0, 255, 0), cv2.FILLED)
    cv2.circle(img, (x2,y2), 15, (0, 255, 0), cv2.FILLED)
    cv2.line(img, (x1,y1), (x2,y2), (255, 0, 255), 3)

    cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)

    length = math.hypot(x2-x1,y2-y1)

    #print(length)

    #hand range 50 - 300
    #volume range -65 - 0

    vol = np.interp(length,[50,300],[minVol, maxVol])
    volBar = np.interp(length,[50,300],[400, 150])
    volPer = np.interp(length,[50,300],[0, 100])
    print(int(length), vol)
    volume.SetMasterVolumeLevel(vol, None)

    if length<=50:
      cv2.circle(img, (cx,cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85, 400), (0,255,0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85,400), (0,255,0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 ,0), 3)



  cTime = time.time()
  fps = 1/(cTime - pTime)
  pTime = cTime

  cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 ,0), 3)


  cv2.imshow("Video",img)
  cv2.waitKey(30)

'''


import cv2
import time
import numpy as np
import HandTracking as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


vc=cv2.VideoCapture(0)

wcam,hcam=640,480
vc.set(3,wcam)
vc.set(4,hcam)
detector=htm.handDetector()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
a=volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)
print(a)
minvol=a[0]
maxvol=a[1]
pTime=0
vol=0
volBar=400

while True:
    success, img=vc.read()
    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)
    if len(lmlist)!=0:
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        length=math.hypot(x1-x2,y1-y2)
        if length<=50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
            vol = np.interp(length,[17,30],[minvol, maxvol])
            volBar = np.interp(length,[17,30],[400, 150])
            volPer = np.interp(length,[17,30],[0, 100])
            print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)
            if length<50:
                cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 ,0), 3)
    cv2.imshow("Krisha",img)
    cv2.waitKey(1)



'''import requests



print("The Question 1: Given some locations say Chennai, Bangalore, Hyderabad, Trivandrum, find the current weather conditions.")


cities = ["Chennai","Bangalore","Hyderabad","Trivandrum"]

apikey="9ikJLyIkcCxCWDr9JlwuMguTnsUO1NDa"
for i in cities:
    URL = "http://dataservice.accuweather.com/locations/v1/cities/search"
    PARAMS = {'apikey':apikey,'q':i}
    r = requests.get(url = URL, params = PARAMS)
    data = r.json()
    loc_key=data[0]['Key']
    URL = "http://dataservice.accuweather.com/currentconditions/v1/"+loc_key
    PARAMS = {'apikey':apikey}
    r = requests.get(url = URL, params = PARAMS)
    data = r.json()
    print("Location id: "+loc_key)
    print("Location: "+i)
    print("Weather : "+data[0]['WeatherText'])
    print("Temperature : "+str(data[0]['Temperature']['Metric']['Value'])+" "+data[0]['Temperature']['Metric']['Unit'])




print("\n\n\nThe Question 2:For the same mentioned locations, list all places that have rained in the last 24 hours.")
URL = "http://dataservice.accuweather.com/locations/v1/cities/search"
location = input("Enter the city name:")
PARAMS = {'apikey':apikey,'q':location}
r = requests.get(url = URL, params = PARAMS)
data = r.json()
print(data)
loc_key=data[0]['Key']

URL = "http://dataservice.accuweather.com/currentconditions/v1/"+loc_key+"/historical/24"
PARAMS = {'apikey':apikey,'details':"true"}
r = requests.get(url = URL, params = PARAMS)
data = r.json()
rainy=[]
for i in data:
    if i['WeatherText'] == 'Rainy':
        rainy.append = i[0][LocalizedName]
print(rainy)
# print("Weather : "+data[0]['WeatherText'])
# print("Temperature : "+str(data[0]['Temperature']['Metric']['Value'])+" "+data[0]['Temperature']['Metric']['Unit'])'''

#select * from Users age greater than 20
