import cv2
import mediapipe as mp
import numpy as np
from math import pi

cam = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw =  mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mpdraw.DrawingSpec(thickness=1, circle_radius=1)
mp_face_mesh = mp.solutions.face_mesh
face = mp_face_mesh.FaceMesh()
axisX = np.zeros(21)
axisY = np.zeros(21)
fingers = np.array([[2,3,4],[5,6,8],[9,11,12],[13,15,16],[17,19,20]]) #polegar, indicador, dedo do meio, anelar e midinho

def calculate_finger_angle(array, id):
    #At first, lets find the equations ax+by+c=0
    #print(axisY[int(array[0])])
    a1 = abs(axisY[array[0]] - axisY[array[1]])
    b1 = abs(axisX[array[0]] - axisX[array[1]])
    m1 = -a1/b1
    a2 = abs(axisY[array[1]] - axisY[array[2]])
    b2 = abs(axisX[array[1]] - axisX[array[2]])
    m2 = -a2/b2
    tgangle = abs((m1-m2)/(1+m1*m2))
    #print(tgangle)
    angle = np.tan(tgangle)
    angle = abs(angle*(180)/(pi))
    #print(angle)
    if id == 3 or id == 4:
        print(angle, id)      
    
    return angle

def meme1(): # frog
    if (axisX[4] <= axisX[8]+9 and axisX[4] >= axisX[8]-9) and axisX[4] != 0 and (axisY[4] <= axisY[8]+10 and axisY[4] >= axisY[8]-10) and axisY[4] != 0:
        if calculate_finger_angle(fingers[2], 2) < 20 and calculate_finger_angle(fingers[3], 3) < 20 and calculate_finger_angle(fingers[4], 4) < 20:
            meme = "Gallery/meme1.png"
            img_meme = cv2.imread(meme)
            cv2.imshow("Meme", img_meme)

def meme2and3(): # billie and finger down
    if calculate_finger_angle(fingers[1], 1) < 20 and calculate_finger_angle(fingers[2], 2) < 20:#Checks if the index finger and the middle finger are straight
        if calculate_finger_angle(fingers[3], 3) > 100 and calculate_finger_angle(fingers[4], 4) > 100: # checks if the others are curved
            print(axisY[5], "5", axisY[8], "8")
            if axisY[8] > axisY[5] and axisY[12] > axisY[9]: # checks if the finger is down
                meme = "Gallery/meme3.png"    
            else: 
                meme = "Gallery/meme2.png" 
            img_meme = cv2.imread(meme)
            cv2.imshow("Meme", img_meme)
        elif axisY[8] > axisY[5] and axisY[12] > axisY[9]:
            meme = "Gallery/meme3.png"
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
        results_hands = hands.process(imgRGB)
        results_face = face.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if results_hands.multi_hand_landmarks:
            for handsl in results_hands.multi_hand_landmarks:
                mpdraw.draw_landmarks(img, handsl, mphands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                for id, lm in enumerate(handsl.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    axisX[id] = cx
                    axisY[id] = cy
            meme1()
            meme2and3()
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mpdraw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        cv2.imshow("Video", img)
        k = cv2.waitKey(1)
        if k%256 == 27: # Leaves with ESC
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
    cam.release()