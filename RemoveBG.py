import  cv2
import numpy as np
import mediapipe as mp
import os
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
# imgBg = cv2.imread("view/3.jpg")

listImag = os.listdir("view")
imgList = []


for imgPath in listImag:
    frame = cv2.imread(f'view/{imgPath}')
    imgList.append(frame)

indexImg = 0

while True:
    sucess, frame = cap.read()
    imgOut = segmentor.removeBG(frame, imgList[indexImg], threshold=0.8)
    imgStacked = cvzone.stackImages([frame, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked, color=(255,0,0))

    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord("a"):
        if indexImg>0:
            indexImg -= 1

    elif key == ord("d"):
        if indexImg<len(imgList)-1:
            indexImg += 1

    elif key == ord("e"):
        break


cap.release()
cv2.destroyAllWindows()


