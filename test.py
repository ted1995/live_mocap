import cv2
import mediapipe as mp
import numpy as np
import socket

joint_list = [[18, 17, 0],[19, 18, 17],[20, 19, 18],
              [14,13,0],[15,14,13],[16,15,14],
              [10,9,0],[11,10,9],[12,11,10],
              [6,5,0],[7,6,5],[8,7,6],
              [2,1,0],[3,2,1],[4,3,2]]  # 手指关节序列

cap = cv2.VideoCapture(0)  # 打开视频文件

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

while cap.isOpened():
    ret, img = cap.read()  # 读取视频帧
    if not ret:
        break
    
    # img= cv2.imread("test1.png")
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img)  # 进行手部关键点检测
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    

    # 绘制手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Keypoints', img)

    # get left hand keypoints
    if results.multi_handedness is None:
        continue

    num_hands_detected = len(results.multi_handedness)
    left_hand_id = list(filter(lambda i: results.multi_handedness[i].classification[0].label == 'Left', range(num_hands_detected)))
    if len(left_hand_id) > 0:
        left_hand_id = left_hand_id[0]
        # world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[left_hand_id].landmark])
        # x = world_landmarks[4] - world_landmarks[3]
        # y = world_landmarks[3] - world_landmarks[2]
        # index3 = caculate_angle_3d(x,y)
        # x = world_landmarks[3] - world_landmarks[2]
        # y = world_landmarks[2] - world_landmarks[1]
        # index2 = caculate_angle_3d(x,y)
        # x = world_landmarks[2] - world_landmarks[1]
        # y = world_landmarks[1] - world_landmarks[0]
        # index1 = caculate_angle_3d(x,y)

        RHL = results.multi_hand_landmarks[left_hand_id]

        jointsAngle = []
        # 计算角度
        for joint in joint_list:
            a = np.array([RHL.landmark[joint[0]].x, RHL.landmark[joint[0]].y])
            b = np.array([RHL.landmark[joint[1]].x, RHL.landmark[joint[1]].y])
            c = np.array([RHL.landmark[joint[2]].x, RHL.landmark[joint[2]].y])
            # 计算弧度
            radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

            if angle > 180.0:
                angle = 360 - angle

            cv2.putText(img, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            jointsAngle.append(angle)
        
        print(jointsAngle)

        jointsAngle = [str(x) for x in jointsAngle]

        res = ",".join(jointsAngle)

        socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        server_add = ("127.0.0.1",5000)
        socket_send.sendto(res.encode('utf-8'),server_add)


        #print("left-1:%s,2:%s,3:%s" % (index1,index2,index3))

    right_hand_id = list(filter(lambda i: results.multi_handedness[i].classification[0].label == 'Right', range(num_hands_detected)))
    if len(right_hand_id) > 0:
        right_hand_id = right_hand_id[0]
        # world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[right_hand_id].landmark])
        # x = world_landmarks[20] - world_landmarks[19]
        # y = world_landmarks[18] - world_landmarks[19]
        # index3 = caculate_angle_3d(x,y)
        # x = world_landmarks[19] - world_landmarks[18]
        # y = world_landmarks[17] - world_landmarks[18]
        # index2 = caculate_angle_3d(x,y)
        # x = world_landmarks[18] - world_landmarks[17]
        # y = world_landmarks[0] - world_landmarks[17]
        # index1 = caculate_angle_3d(x,y)
        # print("right-1:%s,2:%s,3:%s" % (index1,index2,index3))

        LHL = results.multi_hand_landmarks[right_hand_id]

        jointsAngle = []
        # 计算角度
        for joint in joint_list:
            a = np.array([LHL.landmark[joint[0]].x, LHL.landmark[joint[0]].y])
            b = np.array([LHL.landmark[joint[1]].x, LHL.landmark[joint[1]].y])
            c = np.array([LHL.landmark[joint[2]].x, LHL.landmark[joint[2]].y])
            # 计算弧度
            radians_fingers = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians_fingers * 180.0 / np.pi)  # 弧度转角度

            if angle > 180.0:
                angle = 360 - angle

            cv2.putText(img, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            jointsAngle.append(angle)
        
        print(jointsAngle)

        jointsAngle = [str(x) for x in jointsAngle]

        res = ",".join(jointsAngle)

        socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        server_add = ("127.0.0.1",5000)
        socket_send.sendto(res.encode('utf-8'),server_add)


hands.close()  # 关闭手部关键点检测模型
cap.release()  # 释放视频文件
cv2.destroyAllWindows()  # 关闭窗口