import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw =  mp.solutions.drawing_utils

while 1:
    success, img = cam.read()
    if not success:
        print("Failed")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handsl in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handsl, mphands.HAND_CONNECTIONS)

    cv2.imshow("Video", img)
    k = cv2.waitKey(1)
    if k%256 == 27: # Leaves with ESC
        break

cv2.destroyAllWindows()