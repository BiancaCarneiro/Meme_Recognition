import cv2
import mediapipe as mp
import numpy as np

cam = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw =  mp.solutions.drawing_utils
axisX = np.zeros(20)
axisY = np.zeros(20)
fingers = np.array([[2,3,4],[5,6,8],[9,11,12],[13,15,16],[17,19,20]]) #polegar, indicador, dedo do meio, anelar e midinho

def calculate_finger_angle(array):
    #At first, lets find the equations ax+by+c=0
    a1 = axisY[array[0]] - axisY[array[1]]
    b1 = axisX[array[0]] - axisX[array[1]]
    m1 = -a1/b1
    a2 = axisY[array[1]] - axisY[array[2]]
    b2 = axisX[array[1]] - axisX[array[2]]
    m2 = -a2/b2
    tgangle = abs((m1-m2)/(1+m1*m2))
    angle = np.tan(tgangle)
    return angle

def meme1():
    if (axisX[4] <= axisX[8]+8 and axisX[4] >= axisX[8]-8) and axisX[4] != 0 and (axisY[4] <= axisY[8]+8 and axisY[4] >= axisY[8]-8) and axisY[4] != 0:               
        meme1 = "Gallery/meme1.png"
        img_meme1 = cv2.imread(meme1)
        cv2.imshow("Meme", img_meme1)

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
                    if id == 4 or id == 8:
                        #cv2.circle(img, (cx, cy), 15, (255,0,0), cv2.FILLED)
                        axisX[id] = cx
                        axisY[id] = cy
                        print(id, cx, cy)
            meme1()
        cv2.imshow("Video", img)
        k = cv2.waitKey(1)
        if k%256 == 27: # Leaves with ESC
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    main()