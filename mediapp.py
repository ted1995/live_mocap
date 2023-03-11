import mediapipe as mp

import cv2
import time
import socket
import json
import numpy as np

from body_keypoint_track import BodyKeypointTrack



def landmarks_list_to_array(landmark_list):
    array = []
    for lmk in landmark_list.landmark:
        new_row = [lmk.x,lmk.y,lmk.z]
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

cap = cv2.VideoCapture("video/anni.mp4")

# frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_rate = cap.get(cv2.CAP_PROP_FPS)
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# FOV = np.pi / 3
# body_keypoint_track = BodyKeypointTrack(
#     model_complexity=2,
#     im_width=frame_width,
#     im_height=frame_height,
#     fov=FOV,
#     frame_rate=frame_rate,
#     track_hands=False,
#     smooth_range=30 * (1 / frame_rate), #关键点平滑最近30帧
#     smooth_range_barycenter=30 * (1 / frame_rate), #质心平滑最近30帧
# )

mp_pose = mp.solutions.pose.Pose(
    model_complexity=2 , 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

frame_t = 0.0

while True:
    success, image = cap.read()
    image.flags.writeable = False
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # # Get the body keypoints
    # body_keypoint_track.track(frame, frame_t)
    # pose_landmarks1, valid = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)

    # pose_landmarks = []
    # for pose in pose_landmarks1:
    #     pose_landmarks.append(pose.tolist())

    # for pose in  pose_landmarks:
    #     pose[0] = pose[0]*frame_width
    #     pose[1] = -pose[1]*frame_height
    #     pose[2] = -pose[2]

    results = mp_pose.process(image)
    
    h = image.shape[0]
    w = image.shape[1]

    pose_landmarks = []
    if not results.pose_world_landmarks:
        continue
    for lmk in results.pose_world_landmarks.landmark:
        new_row = [lmk.x*w,h-lmk.y*h,lmk.z*w]
        pose_landmarks.append(new_row)

    # ##{0-'hip', 1-'RHip', 2-'RKnee', 3-'RFoot', 4-'LHip',5-'LKnee', 6-'LFoot', 
    #     7-'Spine', 8-'Neck', 9-'Head', 10-'LShoulder', 11-'LElbow', 12-'LWrist', 
    #     13-'RShoulder', 14-'RElbow', 15-'RWrist'}##
    bones_pos = []

    hip = [(pose_landmarks[23][i]+pose_landmarks[24][i])/2.0 for i in range(0,3)]
    bones_pos.append(hip)

    RHip = pose_landmarks[24]
    bones_pos.append(RHip)
    RKnee = pose_landmarks[26]
    bones_pos.append(RKnee)
    RFoot = pose_landmarks[28]
    bones_pos.append(RFoot)

    LHip = pose_landmarks[23]
    bones_pos.append(LHip)
    LKnee = pose_landmarks[25]
    bones_pos.append(LKnee)
    LFoot = pose_landmarks[27]
    bones_pos.append(LFoot)

    Spine = [(pose_landmarks[23][i]+pose_landmarks[24][i]+pose_landmarks[11][i]+pose_landmarks[12][i])/4.0 for i in range(0,3)]
    bones_pos.append(Spine)

    Neck = [(pose_landmarks[9][i]+pose_landmarks[10][i]+pose_landmarks[11][i]+pose_landmarks[12][i])/4.0 for i in range(0,3)]
    bones_pos.append(Neck)

    Head = [(pose_landmarks[0][i]+pose_landmarks[7][i]+pose_landmarks[8][i])/3.0 for i in range(0,3)]
    bones_pos.append(Head)

    LShoulder = pose_landmarks[11]
    bones_pos.append(LShoulder)
    LElbow = pose_landmarks[13]
    bones_pos.append(LElbow)
    LWrist = pose_landmarks[15]
    bones_pos.append(LWrist)

    RShoulder = pose_landmarks[12]
    bones_pos.append(RShoulder)
    RElbow = pose_landmarks[14]
    bones_pos.append(RElbow)
    RWrist = pose_landmarks[16]
    bones_pos.append(RWrist)

    
    strs = []
    for p in bones_pos:
        str1 = ",".join([str(p[0]),str(p[1]),str(p[2])])
        strs.append(str1)
    res = ",".join(strs)
    socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    server_add = ("127.0.0.1",5000)
    socket_send.sendto(res.encode('utf-8'),server_add)

    # frame_t += 1.0 / frame_rate

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

    # strs = []
    # for i in pose_landmarks:
    #     str1 = ",".join([str(i["x"]),str(i["y"]),str(i["z"]),str(i["visibility"])])
    #     strs.append(str1)
    # res = ",".join(strs)
    # socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    # server_add = ("127.0.0.1",5000)
    # socket_send.sendto(res.encode('utf-8'),server_add)
    





