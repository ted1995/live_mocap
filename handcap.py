def detect_hand(img):
    # Find the hand and its landmarks
    hands, img = hand_detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    jointsAngle = []
    if hands:
        for hand in hands:
            handType = hand["type"]  # Handtype Left or Right
            if handType == 'Left':
                jointsAngle.append('Left')
                lmList = hand["lmList"]  # List of 21 Landmark points
                # 计算角度
                for joint in joint_list:
                    a = np.array([lmList[joint[0]][0], lmList[joint[0]][1]])
                    b = np.array([lmList[joint[1]][0], lmList[joint[1]][1]])
                    c = np.array([lmList[joint[2]][0], lmList[joint[2]][1]])
                    # 计算弧度
                    radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

                    if angle > 180.0:
                        angle = 360 - angle

                    cv2.putText(img, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    jointsAngle.append(str(angle))
                
                print(jointsAngle)
            else:
                jointsAngle.append('Right')
                lmList = hand["lmList"]  # List of 21 Landmark points
                # 计算角度
                for joint in joint_list:
                    a = np.array([lmList[joint[0]][0], lmList[joint[0]][1]])
                    b = np.array([lmList[joint[1]][0], lmList[joint[1]][1]])
                    c = np.array([lmList[joint[2]][0], lmList[joint[2]][1]])
                    # 计算弧度
                    radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

                    if angle > 180.0:
                        angle = 360 - angle

                    cv2.putText(img, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    jointsAngle.append(str(angle))
                
                print(jointsAngle)
        return jointsAngle