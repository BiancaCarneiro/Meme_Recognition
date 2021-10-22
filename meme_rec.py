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
#mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
#face = mp_face_mesh.FaceMesh()
axisX = np.zeros(21)
axisY = np.zeros(21)
axisZ = np.zeros(21)
pose_axisX = np.zeros(33)
pose_axisY = np.zeros(33)
pose_axisZ = np.zeros(33)
#face_axisX = np.zeros(4)
#face_axisY = np.zeros(4)
#face_axisZ = np.zeros(4)
fingers = np.array([[2,3,4],[5,6,8],[9,11,12],[13,15,16],[17,19,20]]) #polegar, indicador, dedo do meio, anelar e midinho
arms =  np.array([[12, 14, 16], [12, 14, 16]])
#fingers = np.array([[2,3,4],[5,6,8],[9,11,12],[13,15,16],[17,19,20]]) #polegar, indicador, dedo do meio, anelar e midinho


def calculate_finger_angle(array, id, axisX, axisY, axisZ):
    #At first, lets find the equations ax+by+c=0
    #print(axisY[int(array[0])])
    vectorAB = np.array([axisX[array[0]] - axisX[array[1]], axisY[array[0]] - axisY[array[1]], axisZ[array[0]] - axisZ[array[1]]])
    vectorBC = np.array([axisX[array[1]] - axisX[array[2]], axisY[array[1]] - axisY[array[2]], axisZ[array[1]] - axisZ[array[2]]])
    ABdotBC = np.vdot(vectorAB, vectorBC)
    normAB = np.linalg.norm(vectorAB)
    normBC = np.linalg.norm(vectorBC)
    angle = np.arccos(ABdotBC/(normAB*normBC))
    angle = abs(angle*(180)/(pi))
    #print(angle)
    if id == 3 or id == 4:
        print(angle, id)      
    
    return angle

def meme1(): # frog
    if (axisX[4] <= axisX[8]+9 and axisX[4] >= axisX[8]-9) and axisX[4] != 0 and (axisY[4] <= axisY[8]+10 and axisY[4] >= axisY[8]-10) and axisY[4] != 0:
        if calculate_finger_angle(fingers[2], 2, axisX, axisY, axisZ) < 20 and calculate_finger_angle(fingers[3], 3, axisX, axisY, axisZ) < 20 and calculate_finger_angle(fingers[4], 4, axisX, axisY, axisZ) < 20:
            meme = "Gallery/meme1.png"
            img_meme = cv2.imread(meme)
            cv2.imshow("Meme", img_meme)

def meme2and3(): # billie and finger down
    if calculate_finger_angle(fingers[1], 1, axisX, axisY, axisZ) < 20 and calculate_finger_angle(fingers[2], 2, axisX, axisY, axisZ) < 20:#Checks if the index finger and the middle finger are straight
        if calculate_finger_angle(fingers[3], 3, axisX, axisY, axisZ) > 100 and calculate_finger_angle(fingers[4], 4, axisX, axisY, axisZ) > 100: # checks if the others are curved
            print(axisY[5], "5", axisY[8], "8")
            if axisY[8] > axisY[5] and axisY[12] > axisY[9]: # checks if the finger is down
                meme = "Gallery/meme3.png"    
            else: 
                meme = "Gallery/meme2.png" 
            img_meme = cv2.imread(meme)
            cv2.imshow("Meme", img_meme)

def meme4():
    if (axisX[0] > face_axisX[3] and axisX[0] < face_axisX[1]) or (axisX[0] > face_axisX[1] and axisX[0] < face_axisX[0]):
        meme = "Gallery/meme4.png"
        img_meme = cv2.imread(meme)
        cv2.imshow("Meme", img_meme)

def main():
    while 1:
        success, img = cam.read()
        if not success:
            print("Failed")
            break
        height, width, c = img.shape

        img.flags.writeable = False
        imgBGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(imgBGR)
        results_pose = pose.process(imgBGR)
        img.flags.writeable = True

        #print(results.multi_hand_landmarks)
        if results_pose.pose_landmarks:
            mpdraw.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            for id, lm in enumerate(results_pose.pose_landmarks.landmark):
                cx, cy, cz = int(lm.x * width), int(lm.y * height), int(lm.z)
                pose_axisX[id] = cx
                pose_axisY[id] = cy
                pose_axisZ[id] = cz
        if results_hands.multi_hand_landmarks:
            for handsl in results_hands.multi_hand_landmarks:
                mpdraw.draw_landmarks(img, handsl, mphands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                for id, lm in enumerate(handsl.landmark):
                    cx, cy, cz = int(lm.x * width), int(lm.y * height), int(lm.z)
                    axisX[id] = cx
                    axisY[id] = cy
                    axisZ[id] = cz
            meme1()
            meme2and3()
        
        cv2.imshow("Video", img)
        k = cv2.waitKey(1)
        if k%256 == 27: # Leaves with ESC
            break

    cv2.destroyAllWindows()
    cam.release()

if __name__=='__main__':
    main()