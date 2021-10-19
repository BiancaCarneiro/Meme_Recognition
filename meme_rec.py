import cv2
import mediapipe as mp
import numpy as np
from math import pi

cam = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw =  mp.solutions.drawing_utils
axisX = np.zeros(21)
axisY = np.zeros(21)
fingers = np.array([[2,3,4],[5,6,8],[9,11,12],[13,15,16],[17,19,20]]) #polegar, indicador, dedo do meio, anelar e midinho

def calculate_finger_angle(array, id):
    #At first, lets find the equations ax+by+c=0
    #print(axisY[int(array[0])])
    a1 = axisY[array[0]] - axisY[array[1]]
    b1 = axisX[array[0]] - axisX[array[1]]
    m1 = -a1/b1
    a2 = axisY[array[1]] - axisY[array[2]]
    b2 = axisX[array[1]] - axisX[array[2]]
    m2 = -a2/b2
    tgangle = abs((m1-m2)/(1+m1*m2))
    #print(tgangle)
    angle = np.tan(tgangle)
    angle = abs(angle*(180)/(pi))
    #print(angle)
    if id == 3 or id == 4:
        print(angle, id)
    return angle

def meme1():
    if (axisX[4] <= axisX[8]+9 and axisX[4] >= axisX[8]-9) and axisX[4] != 0 and (axisY[4] <= axisY[8]+10 and axisY[4] >= axisY[8]-10) and axisY[4] != 0:
        if calculate_finger_angle(fingers[2], 2) < 20 and calculate_finger_angle(fingers[3], 3) < 20 and calculate_finger_angle(fingers[4], 4) < 20:
            meme = "Gallery/meme1.png"
            img_meme = cv2.imread(meme)
            cv2.imshow("Meme", img_meme)

def meme2():
    if calculate_finger_angle(fingers[1], 1) < 20 and calculate_finger_angle(fingers[2], 2) < 20:#Checks if the index finger and the middle finger are up
        if calculate_finger_angle(fingers[3], 3) > 100 and calculate_finger_angle(fingers[4], 4) > 100:
            meme = "Gallery/meme2.png"
            img_meme = cv2.imread(meme)
            cv2.imshow("Meme", img_meme)


def main():
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
                    axisX[id] = cx
                    axisY[id] = cy
            meme1()
            meme2()
        cv2.imshow("Video", img)
        k = cv2.waitKey(1)
        if k%256 == 27: # Leaves with ESC
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    main()