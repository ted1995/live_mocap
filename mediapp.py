import mediapipe as mp

import cv2
import time
import socket
import json

# root,spine,chest,neck,lcollar,lshld,lbow,lhand,rcollar,rshld,rbow,rhand,lhip,lknee,lfoot,ltoe,rhip,rknee,rfoot,rtoe,



def landmarks_list_to_array(landmark_list):
    array = []
    for lmk in landmark_list.landmark:
        new_row = {
          'x': lmk.x,
          'y': lmk.y,
          'z': lmk.z,
          'visibility': lmk.visibility
        }
        array.append(new_row)
    return array

def add_extra_points(landmark_list):
    left_shoulder = landmark_list[11]
    right_shoulder = landmark_list[12]
    left_hip = landmark_list[23]
    right_hip = landmark_list[24]

    # Calculating hip position and visibility
    hip = {
          'x': (left_hip['x'] + right_hip['x']) / 2.0,
          'y': (left_hip['y'] + right_hip['y']) / 2.0,
          'z': (left_hip['z'] + right_hip['z']) / 2.0,
          'visibility': (left_hip['visibility'] + right_hip['visibility']) / 2.0
        }
    landmark_list.append(hip)

    # Calculating spine position and visibility
    spine = {
          'x': (left_hip['x'] + right_hip['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
          'y': (left_hip['y'] + right_hip['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
          'z': (left_hip['z'] + right_hip['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0,
          'visibility': (left_hip['visibility'] + right_hip['visibility'] + right_shoulder['visibility'] + left_shoulder['visibility']) / 4.0
        }
    landmark_list.append(spine)

    left_mouth = landmark_list[9]
    right_mouth = landmark_list[10]
    nose = landmark_list[0]
    left_ear = landmark_list[7]
    right_ear = landmark_list[8]
    # Calculating neck position and visibility
    neck = {
          'x': (left_mouth['x'] + right_mouth['x'] + right_shoulder['x'] + left_shoulder['x']) / 4.0,
          'y': (left_mouth['y'] + right_mouth['y'] + right_shoulder['y'] + left_shoulder['y']) / 4.0,
          'z': (left_mouth['z'] + right_mouth['z'] + right_shoulder['z'] + left_shoulder['z']) / 4.0,
          'visibility': (left_mouth['visibility'] + right_mouth['visibility'] + right_shoulder['visibility'] + left_shoulder['visibility']) / 4.0
        }
    landmark_list.append(neck)

    # Calculating head position and visibility
    head = {
          'x': (nose['x'] + left_ear['x'] + right_ear['x']) / 3.0,
          'y': (nose['y'] + left_ear['y'] + right_ear['y']) / 3.0,
          'z': (nose['z'] + left_ear['z'] + right_ear['z']) / 3.0,
          'visibility': (nose['visibility'] + left_ear['visibility'] + right_ear['visibility']) / 3.0,
        }
    landmark_list.append(head)

cap = cv2.VideoCapture("video/ikun.mp4")

mp_pose = mp.solutions.pose.Pose(
    model_complexity=2 , 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

while True:
    success, image = cap.read()
    time.sleep(0.01)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(image)
    

    pose_landmarks = landmarks_list_to_array(results.pose_world_landmarks)
    rows, cols, _ = image.shape
    add_extra_points(pose_landmarks)

    # poseba = []
    # poseba.append(pose_landmarks[12])
    # poseba.append(pose_landmarks[14])
    # poseba.append(pose_landmarks[16])
    # poseba.append(pose_landmarks[22])
    # poseba.append(pose_landmarks[20])

    # poseba.append(pose_landmarks[11])
    # poseba.append(pose_landmarks[13])
    # poseba.append(pose_landmarks[15])
    # poseba.append(pose_landmarks[21])
    # poseba.append(pose_landmarks[19])

    # poseba.append(pose_landmarks[7])
    # poseba.append(pose_landmarks[2])
    # poseba.append(pose_landmarks[8])
    # poseba.append(pose_landmarks[5])
    # poseba.append(pose_landmarks[0])

    # poseba.append(pose_landmarks[24])
    # poseba.append(pose_landmarks[26])
    # poseba.append(pose_landmarks[28])
    # poseba.append(pose_landmarks[30])

    # poseba.append(pose_landmarks[23])
    # poseba.append(pose_landmarks[25])
    # poseba.append(pose_landmarks[27])
    # poseba.append(pose_landmarks[29])

    # poseba.append(pose_landmarks[34])
    # poseba.append(pose_landmarks[33])
    # poseba.append(pose_landmarks[36])
    # poseba.append(pose_landmarks[35])
    # poseba.append(pose_landmarks[34])

    # strs = []
    # for i in pose_landmarks:
    #     str1 = ",".join([str(i["x"]),str(i["y"]),str(i["z"])])
    #     strs.append(str1)
    # res = ",".join(strs)
    # socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    # server_add = ("127.0.0.1",5000)
    # socket_send.sendto(res.encode('utf-8'),server_add)

    strs = []
    for i in pose_landmarks:
        str1 = ",".join([str(i["x"]),str(i["y"]),str(i["z"]),str(i["visibility"])])
        strs.append(str1)
    res = ",".join(strs)
    socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    server_add = ("127.0.0.1",5000)
    socket_send.sendto(res.encode('utf-8'),server_add)
    





