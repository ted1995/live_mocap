#####################################################################################
# Single View Human Motion Capture, Based on Mediapipe & OpenCV & PyTorch
# 
# Author: Ruicheng Wang
# License: Apache License 2.0
#####################################################################################
import os
import shutil
import argparse
import pickle
import subprocess
import json
import socket
import mathutils
import math

import numpy as np
from cvzone.HandTrackingModule import HandDetector
import cv2
import torch
from tqdm import tqdm

from body_keypoint_track import BodyKeypointTrack, show_annotation
from skeleton_ik_solver import SkeletonIKSolver

# face detection and facial landmark
from facial_landmark import FaceMeshDetector
# pose estimation and stablization
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
# Miscellaneous detections (eyes/ mouth...)
from facial_features import FacialFeatures, Eyes

joint_list = [[18, 17, 0],[19, 18, 17],[20, 19, 18],
              [14,13,0],[15,14,13],[16,15,14],
              [10,9,0],[11,10,9],[12,11,10],
              [6,5,0],[7,6,5],[8,7,6],
              [2,1,0],[3,2,1],[4,3,2]]  # 手指关节序列

# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]

# for eyes
eyes_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]

# for mouth_dist
mouth_dist_stabilizer = Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1
)

# Facemesh
detector = FaceMeshDetector()

hand_detector = HandDetector(detectionCon=0.5, maxHands=2)

# extra 10 points due to new attention model (in iris detection)
iris_image_points = np.zeros((10, 2))

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
    

def detect_hand1(img):
    # Find the hand and its landmarks
    hands, img = hand_detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    str_right = ""
    str_left = ""
    if hands:
        for hand in hands:
            handType = hand["type"]  # Handtype Left or Right
            if handType == 'Left':
                lmlist = hand["lmList"]  # List of 21 Landmark points
                lmlist = [str(i) for x in lmlist for i in x]
                str_left = "Left," + ",".join(lmlist)
            else:
                lmlist = hand["lmList"]  # List of 21 Landmark points
                lmlist = [str(i) for x in lmlist for i in x]
                str_right = "Right," + ",".join(lmlist)
    if str_left and str_right:
        return str_left + "," + str_right
    elif str_left and not str_right:
        return str_left
    elif not str_left and str_right:
        return str_right
    else:
        return ""


def detect_face(img,image_points,pose_estimator):
    # first two steps
    img_facemesh, faces = detector.findFaceMesh(img)

    # if there is any face detected
    if faces:
        # only get the first face
        for i in range(len(image_points)):
            image_points[i, 0] = faces[0][i][0]
            image_points[i, 1] = faces[0][i][1]
            
        # for refined landmarks around iris
        for j in range(len(iris_image_points)):
            iris_image_points[j, 0] = faces[0][j + 468][0]
            iris_image_points[j, 1] = faces[0][j + 468][1]

        # The third step: pose estimation
        # pose: [[rvec], [tvec]]
        pose = pose_estimator.solve_pose_by_all_points(image_points)

        x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)
        x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.RIGHT)


        ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
        ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

        pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

        mar = FacialFeatures.mouth_aspect_ratio(image_points)
        mouth_distance = FacialFeatures.mouth_distance(image_points)

        # Stabilize the pose.
        steady_pose = []
        pose_np = np.array(pose).flatten()

        for value, ps_stb in zip(pose_np, pose_stabilizers):
            ps_stb.update([value])
            steady_pose.append(ps_stb.state[0])

        steady_pose = np.reshape(steady_pose, (-1, 3))

        # stabilize the eyes value
        steady_pose_eye = []
        for value, ps_stb in zip(pose_eye, eyes_stabilizers):
            ps_stb.update([value])
            steady_pose_eye.append(ps_stb.state[0])

        mouth_dist_stabilizer.update([mouth_distance])
        steady_mouth_dist = mouth_dist_stabilizer.state[0]

        roll = np.clip(steady_pose[0][1], -np.pi/2, np.pi/2)
        if np.degrees(steady_pose[0][0])<0:
            pitch = np.clip(-(np.pi + steady_pose[0][0]), -np.pi/2, np.pi/2)
            yaw =  -np.clip(steady_pose[0][2], -np.pi/2, np.pi/2)
        else:
            pitch = np.clip(-(-np.pi + steady_pose[0][0]), -np.pi/2, np.pi/2)
            yaw =  np.clip(steady_pose[0][2], -np.pi/2, np.pi/2)
        print(np.degrees(steady_pose[0][0]))
        print("Roll: %.2f, Pitch: %.2f, Yaw: %.2f" % (roll, pitch, yaw))

        return roll,pitch,yaw



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=str, help='Path to rigged model Blender file. eg. c:\\tmp\\model.blend')
    parser.add_argument('--video', type=str, help='Path to video file. eg. c:\\tmp\\video.mp4')
    parser.add_argument('--track_hands', action='store_true', help='Enable hand tracking')

    args = parser.parse_args()
    FOV = np.pi / 3

    # 导出blend中人物模型的骨骼样式
    os.makedirs('tmp', exist_ok=True)
    print("Export skeleton...")
    if os.path.exists('tmp/skeleton'):
        shutil.rmtree('tmp/skeleton')
    os.system(f"blender {args.blend} --background --python export_skeleton.py")
    if not os.path.exists('tmp/skeleton'):
        raise Exception("Skeleton export failed")
    
    # 手动更新骨骼映射
    # with open("tmp/skeleton/skeleton.json",'r') as f:
    #     js = json.load(f)
    # with open("model/%s.json" % args.blend.split("/")[-1].split(".")[0],'r') as f:
    #     js1 = json.load(f)
        
    # js['bone_remap'] = js1["bone_remap"]

    # with open("tmp/skeleton/skeleton.json",'w') as f:
    #     f.write(json.dumps(js,indent=4))

    # Open the video capture
    cap = cv2.VideoCapture(args.video)

    
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Pose estimation related
    pose_estimator = PoseEstimator((frame_height, frame_width))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))

 
    if not cap.isOpened():
        raise Exception("Video capture failed")
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the body keypoint tracker
    body_keypoint_track = BodyKeypointTrack(
        model_complexity=1,
        im_width=frame_width,
        im_height=frame_height,
        fov=FOV,
        frame_rate=frame_rate,
        track_hands=False,
        smooth_range=10 * (1 / frame_rate), #关键点平滑最近30帧
        smooth_range_barycenter=10 * (1 / frame_rate), #质心平滑最近30帧
    )

    # Initialize the skeleton IK solver
    skeleton_ik_solver = SkeletonIKSolver(
        model_path='tmp/skeleton',
        track_hands=False,
        max_iter = 20, #最大迭代次数，表示LBFGS优化器最多迭代的次数。建议将最大迭代次数设置为20到100之间。目前测试无论设置多大，都只会迭代20次。
        lr = 1, #学习率，表示每次迭代更新的步长。LBFGS优化器不需要学习率调度器，因此只需要指定一个固定的学习率即可。建议将学习率设置为1，即默认值。
        tolerance_change = 1e-9, #变化容差，表示LBFGS优化器停止迭代的变化容差。建议将变化容差设置为1e-9到1e-12之间。
        tolerance_grad = 1e-5, #梯度容差，表示LBFGS优化器停止迭代的梯度容差。建议将梯度容差设置为1e-5到1e-9之间。
        joint_constraint_loss_weight = 1e-1,
        pose_reg_loss_weight = 1e-2,   
        smooth_range=10 * (1 / frame_rate), #MLS平滑的范围，可以平滑人物运动，但是设置太大会吃动作。
    )

    bone_euler_sequence, scale_sequence, location_sequence = [], [], []

    frame_t = 0.0
    frame_i = 0
    bar = tqdm(total=total_frames, desc='Running...')
    while cap.isOpened():
        # Get the frame image
        ret, frame = cap.read()

        if not ret:
            break

        img = frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the body keypoints
        body_keypoint_track.track(frame, frame_t)
        kpts3d, valid = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)

        # Solve the skeleton IK
        skeleton_ik_solver.fit(torch.from_numpy(kpts3d).float(), torch.from_numpy(valid).bool(), frame_t)

        # Get the skeleton pose
        bone_euler = skeleton_ik_solver.get_smoothed_bone_euler(frame_t)
        location = skeleton_ik_solver.get_smoothed_location(frame_t)
        scale = skeleton_ik_solver.get_scale()

        bone_euler_sequence.append(bone_euler)
        location_sequence.append(location)
        scale_sequence.append(skeleton_ik_solver.get_scale())

        # Show keypoints tracking result
        show_annotation(frame, kpts3d, valid, body_keypoint_track.K)
        if cv2.waitKey(1) == 27:
            print('Cancelled by user. Exit.')
            exit()

        frame_i += 1
        frame_t += 1.0 / frame_rate
        bar.update(1)
        if frame_i == 370:
            print("comming")
        if frame_i >300:
            break

        strlist = []
        bone_euler = bone_euler.tolist()
        for b in bone_euler:
            # 替换neck的欧拉角
            if bone_euler.index(b) == 1:
                if detect_face(img, image_points,pose_estimator) is not None:
                    roll,pitch,yaw = detect_face(img, image_points,pose_estimator)
                    b = [roll,pitch,yaw]
            strl = ",".join([str(b[i]) for i in range(0,3)])
            strlist.append(strl)
        location = location.tolist()
        print(location)
        str1 = ",".join([str(location[i]) for i in range(0,3)])
        strlist.append(str1)

        # joint_angle = detect_hand(img)
        # if joint_angle:
        #     str1 = ",".join(joint_angle)
        # strlist.append(str1)

        for i in skeleton_ik_solver.optimizable_bones:
            if i == "left_wrist":
                print(i+":"+strlist[skeleton_ik_solver.optimizable_bones.index(i)])

        # tmp = detect_hand1(img)
        # if tmp:
        #     res = ",".join(strlist)+","+tmp
        # else:
        #     
        
        res = ",".join(strlist)
        print(res)

        socket_send = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        server_add = ("127.0.0.1",5000)
        socket_send.sendto(res.encode('utf-8'),server_add)
    
    # print(skeleton_ik_solver.optimizable_bones)
    
    # Save animation result
    print("Save animation result...")
    with open('tmp/bone_animation_data.pkl', 'wb') as fp:
        pickle.dump({
            'fov': FOV,
            'frame_rate': frame_rate,
            'bone_names': skeleton_ik_solver.optimizable_bones,
            'bone_euler_sequence': bone_euler_sequence,
            'location_sequence': location_sequence,
            'scale': np.mean(scale_sequence),
            'all_bone_names': skeleton_ik_solver.all_bone_names
        }, fp)

    # Open blender and apply the animation
    print("Open blender and apply animation...")
    
    proc = subprocess.Popen(f"blender {args.blend} --python apply_animation.py")
    proc.wait()


if __name__ == '__main__':
    main()