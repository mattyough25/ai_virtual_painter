import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[3]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
while True:
    # Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

        # If selection mode (two fingers are up), 
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1,y1-25), (x2,y2+25), (255,0,255),cv2.FILLED)
            if y1 < 126:
                if 250<x1<450:
                    header = overlayList[3]
                elif 600<x1<800:
                    header = overlayList[0]
                elif 1050<x1<1200:
                    header = overlayList[1]
                elif 0<x1<200:
                    header = overlayList[2]

        # If drawing mode (index finger is up)
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, (255,0,255),cv2.FILLED)

    # Setting the header image
    img[0:126, 0:1280] = header
    cv2.imshow("Image", img)
    # Enter key 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
