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
import cv2
import torch
from tqdm import tqdm

from body_keypoint_track import BodyKeypointTrack, show_annotation
from skeleton_ik_solver import SkeletonIKSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=str, help='Path to rigged model Blender file. eg. c:\\tmp\\model.blend')
    parser.add_argument('--video', type=str, help='Path to video file. eg. c:\\tmp\\video.mp4')
    parser.add_argument('--track_hands', action='store_true', help='Enable hand tracking')

    args = parser.parse_args()
    FOV = np.pi / 3

    # Call blender to export skeleton
    os.makedirs('tmp', exist_ok=True)
    print("Export skeleton...")
    if os.path.exists('tmp/skeleton'):
        shutil.rmtree('tmp/skeleton')
    os.system(f"blender {args.blend} --background --python export_skeleton.py")
    if not os.path.exists('tmp/skeleton'):
        raise Exception("Skeleton export failed")
    
    # with open("tmp/skeleton/skeleton.json",'r') as f:
    #     js = json.load(f)
    # with open("model/%s.json" % args.blend.split("/")[-1].split(".")[0],'r') as f:
    #     js1 = json.load(f)
        
    # js['bone_remap'] = js1["bone_remap"]

    # with open("tmp/skeleton/skeleton.json",'w') as f:
    #     f.write(json.dumps(js,indent=4))

    # Open the video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise Exception("Video capture failed")
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the body keypoint tracker
    body_keypoint_track = BodyKeypointTrack(
        model_complexity=2,
        im_width=frame_width,
        im_height=frame_height,
        fov=FOV,
        frame_rate=frame_rate,
        track_hands=True,
        smooth_range=10 * (1 / frame_rate), #关键点平滑最近30帧
        smooth_range_barycenter=10 * (1 / frame_rate), #质心平滑最近30帧
    )

    # Initialize the skeleton IK solver
    skeleton_ik_solver = SkeletonIKSolver(
        model_path='tmp/skeleton',
        track_hands=True,
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
        if frame_i >500:
            break

        strlist = []
        for b in bone_euler:
            tb = b.tolist()
            strl = ",".join([str(tb[i]) for i in range(0,3)])
            strlist.append(strl)
        location = location.tolist()
        print(location)
        str1 = ",".join([str(location[i]) for i in range(0,3)])
        strlist.append(str1)
        for i in skeleton_ik_solver.optimizable_bones:
            if i == "left_index1":
                print(i+":"+strlist[skeleton_ik_solver.optimizable_bones.index(i)])

        res = ",".join(strlist)
        

        #print(res)

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