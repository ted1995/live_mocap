import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture("video/baike.mp4")  # 打开视频文件

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
    
    img= cv2.imread("test2.png")
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
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[left_hand_id].landmark])
        x = world_landmarks[20] - world_landmarks[19]
        y = world_landmarks[18] - world_landmarks[19]
        index3 = caculate_angle_3d(x,y)
        x = world_landmarks[19] - world_landmarks[18]
        y = world_landmarks[17] - world_landmarks[18]
        index2 = caculate_angle_3d(x,y)
        x = world_landmarks[18] - world_landmarks[17]
        y = world_landmarks[0] - world_landmarks[17]
        index1 = caculate_angle_3d(x,y)
        print("1:%s,2:%s,3:%s" % (index1,index2,index3))

    right_hand_id = list(filter(lambda i: results.multi_handedness[i].classification[0].label == 'Right', range(num_hands_detected)))
    if len(right_hand_id) > 0:
        right_hand_id = right_hand_id[0]
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[right_hand_id].landmark])
        x = world_landmarks[20] - world_landmarks[19]
        y = world_landmarks[18] - world_landmarks[19]
        index3 = caculate_angle_3d(x,y)
        x = world_landmarks[19] - world_landmarks[18]
        y = world_landmarks[17] - world_landmarks[18]
        index2 = caculate_angle_3d(x,y)
        x = world_landmarks[18] - world_landmarks[17]
        y = world_landmarks[0] - world_landmarks[17]
        index1 = caculate_angle_3d(x,y)
        print("1:%s,2:%s,3:%s" % (index1,index2,index3))


hands.close()  # 关闭手部关键点检测模型
cap.release()  # 释放视频文件
cv2.destroyAllWindows()  # 关闭窗口