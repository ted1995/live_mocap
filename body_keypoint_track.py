import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
from typing import List, Tuple, Dict

import mediapipe as mp
import numpy as np
import cv2

from utils3d import intrinsic_from_fov, mls_smooth_numpy
from cvzone.HandTrackingModule import HandDetector

hand_detector = HandDetector(detectionCon=0.5, maxHands=2)

MEDIAPIPE_POSE_KEYPOINTS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
]   # 33

MEDIAPIPE_HAND_KEYPOINTS = [
    "wrist", "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4"
]   # 21

ALL_KEYPOINTS = MEDIAPIPE_POSE_KEYPOINTS + ['left_' + s for s in MEDIAPIPE_HAND_KEYPOINTS] + ['right_' + s for s in MEDIAPIPE_HAND_KEYPOINTS]

MEDIAPIPE_POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]

WEIGHTS = {
    'left_ear': 0.04,
    'right_ear': 0.04,
    'left_shoulder': 0.18,
    'right_shoulder': 0.18,
    'left_elbow': 0.02,
    'right_elbow': 0.02,
    'left_wrist': 0.01,
    'right_wrist': 0.01,
    'left_hip': 0.2,
    'right_hip': 0.2,
    'left_knee': 0.03,
    'right_knee': 0.03,
    'left_ankle': 0.02,
    'right_ankle': 0.02,
}


class BodyKeypointTrack:
    def __init__(self, im_width: int, im_height: int, fov: float, frame_rate: float, *, track_hands: bool = True, model_complexity=1, smooth_range: float = 0.3, smooth_range_barycenter: float = 1.0):
        self.K = intrinsic_from_fov(fov, im_width, im_height)
        self.im_width, self.im_height = im_width, im_height
        self.frame_delta = 1. / frame_rate

        self.global_world_landmarks = []

        self.mp_pose_model = mp.solutions.pose.Pose(
            model_complexity=model_complexity, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.pose_rvec, self.pose_tvec = None, None
        self.pose_kpts2d = self.pose_kpts3d = None
        # 关键点对应的权重矩阵
        self.barycenter_weight = np.array([WEIGHTS.get(kp, 0.) for kp in MEDIAPIPE_POSE_KEYPOINTS])

        self.track_hands = track_hands
        if track_hands:
            self.mp_hands_model = mp.solutions.hands.Hands(
                max_num_hands=2,
                model_complexity=min(model_complexity, 1), 


                
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            )
            self.left_hand_rvec, self.left_hand_tvec = None, None
            self.left_hand_kpts2d = self.left_hand_kpts3d = None
            self.right_hand_rvec, self.right_hand_tvec = None, None
            self.right_hand_kpts2d = self.right_hand_kpts3d = None
        
        self.smooth_range = smooth_range
        self.smooth_range_barycenter = smooth_range_barycenter
        self.barycenter_history: List[Tuple[np.ndarray, float]] = []
        self.pose_history: List[Tuple[np.ndarray, float]] = []
        self.left_hand_history: List[Tuple[np.ndarray, float]] = []
        self.right_hand_history: List[Tuple[np.ndarray, float]] = []

    def _get_camera_space_landmarks(self, image_landmarks, world_landmarks, visible, rvec, tvec):
        # get transformation matrix from world coordinate to camera coordinate
        # 计算相机相对于世界坐标系的位姿，输出rvec为旋转向量，tvec为平移向量
        _, rvec, tvec = cv2.solvePnP(world_landmarks[visible], image_landmarks[visible], self.K, np.zeros(5), rvec=rvec, tvec=tvec, useExtrinsicGuess=rvec is not None)
        # 将旋转向量转化为旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)
        
        # get camera coordinate of all keypoints
        # 将所有点转化到相机坐标系中
        kpts3d_cam = world_landmarks @ rmat.T + tvec.T

        # force projected x, y to be identical to visibile image_landmarks
        # 去除所有坐标的z值，转化为n行1列的数组
        kpts3d_cam_z = kpts3d_cam[:, 2].reshape(-1, 1)
        # (np.concatenate([image_landmarks, np.ones((image_landmarks.shape[0], 1))], axis=1) [1,2] -> [1,2,1]

        kpts3d_cam[:, :2] =  (np.concatenate([image_landmarks, np.ones((image_landmarks.shape[0], 1))], axis=1) @ np.linalg.inv(self.K).T * kpts3d_cam_z)[:, :2]
        return kpts3d_cam, rvec, tvec

    def _track_pose(self, image: np.ndarray, t: float):
        self.pose_kpts2d = self.pose_kpts3d = self.barycenter = None

        results = self.mp_pose_model.process(image)

        if results.pose_landmarks is None:
            return 

        image_landmarks = np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in results.pose_landmarks.landmark])
        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
        self.global_world_landmarks = [[-lm.x * self.im_width, self.im_height - lm.y * self.im_height, -lm.z * self.im_width] for lm in results.pose_landmarks.landmark]
        visible = np.array([lm.visibility > 0.2 for lm in results.pose_landmarks.landmark])

        if visible.sum() < 6:
            return 
        kpts3d, rvec, tvec = self._get_camera_space_landmarks(image_landmarks, world_landmarks, visible, self.pose_rvec, self.pose_tvec)
        if tvec[2] < 0:
            return

        self.pose_kpts2d = image_landmarks
        self.barycenter = np.average(kpts3d, axis=0, weights=self.barycenter_weight)
        self.pose_kpts3d = kpts3d - self.barycenter
        self.pose_rvec, self.pose_tvec = rvec, tvec
        self.pose_history.append((kpts3d, t))

    def _track_hands(self, image: np.ndarray, t: float):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hands, img = hand_detector.findHands(image)
        if hands:
            for hand in hands:
                handType = hand["type"]  # Handtype Left or Right
                if handType == 'Left':
                    lmlist = hand["lmList"]  # List of 21 Landmark points
                    self.left_hand_history.append((lmlist,t))
                else:
                    lmlist = hand["lmList"]  # List of 21 Landmark points
                    self.right_hand_history.append((lmlist,t))

    def track(self, image: np.ndarray, frame_t: float):
        self._track_pose(image, frame_t)
        if self.track_hands and self.pose_kpts3d is not None:
            self._track_hands(image, frame_t)

    def get_smoothed_3d_keypoints(self, query_t: float):
        
        # Get smoothed pose keypoints
        pose_kpts3d_list = [kpts3d for kpts3d, t in self.pose_history if abs(t - query_t) < self.smooth_range]
        pose_t = [t for kpts3d, t in self.pose_history if abs(t - query_t) < self.smooth_range]
        pose_kpts3d = None if not any(abs(t - query_t) < self.frame_delta * 0.6  for t in pose_t) else mls_smooth_numpy(pose_t, pose_kpts3d_list, query_t, self.smooth_range)

        all_kpts3d = pose_kpts3d if pose_kpts3d is not None else np.zeros((len(MEDIAPIPE_POSE_KEYPOINTS), 3))
        all_valid = np.full(len(MEDIAPIPE_POSE_KEYPOINTS), pose_kpts3d is not None)

        if self.track_hands:
            # Get smoothed left hand keypoints
            left_hand_kpts3d_list = [kpts3d for kpts3d, t in self.left_hand_history if abs(t - query_t) < self.smooth_range]
            left_hand_t = [t for kpts3d, t in self.left_hand_history if abs(t - query_t) < self.smooth_range]
            if any(abs(t - query_t) < self.frame_delta * 0.6 for t in left_hand_t):
                left_hand_kpts3d = mls_smooth_numpy(left_hand_t, left_hand_kpts3d_list, query_t, self.smooth_range)
            else:
                left_hand_kpts3d = None
                
            # Get smoothed right hand keypoints
            right_hand_kpts3d_list = [kpts3d for kpts3d, t in self.right_hand_history if abs(t - query_t) < self.smooth_range]
            right_hand_t = [t for kpts3d, t in self.right_hand_history if abs(t - query_t) < self.smooth_range]
            if any(abs(t - query_t) < self.frame_delta * 0.6 for t in right_hand_t):
                right_hand_kpts3d = mls_smooth_numpy(right_hand_t, right_hand_kpts3d_list, query_t, self.smooth_range)
            else:
                right_hand_kpts3d = None
            
        # left_hand_kpts3d = left_hand_kpts3d.tolist() if left_hand_kpts3d is not None else None
        # right_hand_kpts3d = right_hand_kpts3d.tolist()  if right_hand_kpts3d is not None else None
        return all_kpts3d, all_valid,left_hand_kpts3d,right_hand_kpts3d


def show_annotation(image, kpts3d, valid, intrinsic):
    annotate_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kpts3d_homo = kpts3d @ intrinsic.T
    kpts2d = kpts3d_homo[:, :2] / kpts3d_homo[:, 2:]
    for a, b in MEDIAPIPE_POSE_CONNECTIONS:
        if valid[a] == 0 or valid[b] == 0:
            continue
        cv2.line(annotate_image, (int(kpts2d[a, 0]), int(kpts2d[a, 1])), (int(kpts2d[b, 0]), int(kpts2d[b, 1])), (0, 255, 0), 1)
    for i in range(kpts2d.shape[0]):
        if valid[i] == 0:
            continue
        cv2.circle(annotate_image, (int(kpts2d[i, 0]), int(kpts2d[i, 1])), 2, (0, 0, 255), -1)
    cv2.imshow('Keypoint annotation', annotate_image)

def test():
    import tqdm

    INPUT_FILE = 'C:\\Users\\16215\\Pictures\\视频项目\\orange.mp4'
    INPUT_IMAGE_SIZE = (640, 360)
    cap = cv2.VideoCapture(INPUT_FILE)
    kpts3ds = []
    
    body_keypoint_track = BodyKeypointTrack(
        *INPUT_IMAGE_SIZE, 
        np.pi / 4, 
        track_hands=False, 
        smooth_range=0.3, 
        smooth_range_barycenter=1.0, 
        frame_delta=1.0 / 30.0
    )

    frame_t = 0.0
    frame_i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), INPUT_IMAGE_SIZE)
        body_keypoint_track.track(frame, frame_t)
        kpts3d, visib = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)
        kpts3ds.append((kpts3d, visib))

        kpts3d_homo = kpts3d @ body_keypoint_track.K.T
        kpts2d = kpts3d_homo[:, :2] / kpts3d_homo[:, 2:]

        annotate_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for a, b in MEDIAPIPE_POSE_CONNECTIONS:
            if visib[a] == 0 or visib[b] == 0:
                continue
            cv2.line(annotate_image, (int(kpts2d[a, 0]), int(kpts2d[a, 1])), (int(kpts2d[b, 0]), int(kpts2d[b, 1])), (0, 255, 0), 1)
        for i in range(kpts2d.shape[0]):
            if visib[i] == 0:
                continue
            cv2.circle(annotate_image, (int(kpts2d[i, 0]), int(kpts2d[i, 1])), 2, (0, 0, 255), -1)
            
        cv2.imshow('annot', annotate_image)
        cv2.imwrite('tmp/tomas/%04d_annot.jpg' % frame_i, annotate_image)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_t += 1/30
        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()
    with open('tmp/kpts3ds_mengnan.pkl', 'wb') as f:
        pickle.dump(kpts3ds, f)

if __name__ == '__main__':
    test()
