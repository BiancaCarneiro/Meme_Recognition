import cv2
import mediapipe as mp
import numpy as np

cam = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw =  mp.solutions.drawing_utils
axisX = np.zeros(20)
axisY = np.zeros(20)

while 1:
    success, img = cam.read()
    if not success:
        print("Failed")
        break
    height, width, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handsl in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handsl, mphands.HAND_CONNECTIONS)
            for id, lm in enumerate(handsl.landmark):
                cx, cy = int(lm.x * width), int(lm.y * height)
                if id == 4 or id == 8:
                    #cv2.circle(img, (cx, cy), 15, (255,0,0), cv2.FILLED)
                    axisX[id] = cx
                    axisY[id] = cy
                    print(id, cx, cy)
    if (axisX[4] <= axisX[8]+6 and axisX[4] >= axisX[8]-6) and axisX[4] != 0 and (axisY[4] <= axisY[8]+6 and axisY[4] >= axisY[8]-6) and axisY[4] != 0:               
        meme1 = "Gallery/meme1.png"
        img_meme1 = cv2.imread(meme1)
        cv2.imshow("Meme 1", img_meme1)

    cv2.imshow("Video", img)
    k = cv2.waitKey(1)
    if k%256 == 27: # Leaves with ESC
        break

cv2.destroyAllWindows()