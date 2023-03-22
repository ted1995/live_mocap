import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture("video/hand.mp4")  # 打开视频文件

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)

while cap.isOpened():
    ret, img = cap.read()  # 读取视频帧
    if not ret:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    results = hands.process(img)  # 进行手部关键点检测

    

    if results.multi_hand_landmarks:  # 如果检测到手部关键点
        for hand_landmarks in results.multi_hand_landmarks:  # 遍历每一只手
            for lm in hand_landmarks.landmark:
                print(lm.visibility )

            for idx, lm in enumerate(hand_landmarks.landmark):  # 遍历每个关键点
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # 计算关键点在图像上的坐标
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)  # 在图像上绘制关键点

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()  # 关闭手部关键点检测模型
cap.release()  # 释放视频文件
cv2.destroyAllWindows()  # 关闭窗口