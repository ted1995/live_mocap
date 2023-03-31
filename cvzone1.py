from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np


def caculate_angle_3d(x,y):
    # 分别计算两个向量的模：
    l_x=np.sqrt(x.dot(x))
    l_y=np.sqrt(y.dot(y))
    # 计算两个向量的点积
    dian=x.dot(y)
    # 计算夹角的cos值：
    cos_=dian/(l_x*l_y)
    # 求得夹角（弧度制）：
    angle_hu=np.arccos(cos_)
    # 转换为角度值：
    angle_d=angle_hu*180/np.pi
    return angle_d

cap = cv2.VideoCapture("video/jackma.mp4")
detector = HandDetector(detectionCon=0.5, maxHands=2)
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        world_landmarks = np.array(lmList1)
        x = world_landmarks[4] - world_landmarks[3]
        y = world_landmarks[2] - world_landmarks[3]
        index3 = caculate_angle_3d(x,y)
        x = world_landmarks[3] - world_landmarks[2]
        y = world_landmarks[1] - world_landmarks[2]
        index2 = caculate_angle_3d(x,y)
        x = world_landmarks[2] - world_landmarks[1]
        y = world_landmarks[0] - world_landmarks[1]
        index1 = caculate_angle_3d(x,y)
        print("%s-1:%s,2:%s,3:%s" % (handType1,index1,index2,index3))

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            world_landmarks = np.array(lmList2)
            x = world_landmarks[20] - world_landmarks[19]
            y = world_landmarks[18] - world_landmarks[19]
            index3 = caculate_angle_3d(x,y)
            x = world_landmarks[19] - world_landmarks[18]
            y = world_landmarks[17] - world_landmarks[18]
            index2 = caculate_angle_3d(x,y)
            x = world_landmarks[18] - world_landmarks[17]
            y = world_landmarks[0] - world_landmarks[17]
            index1 = caculate_angle_3d(x,y)

            print("%s-1:%s,2:%s,3:%s" % (handType2,index1,index2,index3))

            fingers2 = detector.fingersUp(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            #length, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()