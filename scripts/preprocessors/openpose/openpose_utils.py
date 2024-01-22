# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th added to RetroDiffusion

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from .face import Face


def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = util.draw_handpose(canvas, hands)

    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class OpenposeDetector:
    def __init__(self, body_modelpath, hand_modelpath, face_modelpath):
        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def __call__(self, oriImg, hand_and_face=False, return_is_index=False):
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            hands = []
            faces = []
            if hand_and_face:
                # Hand
                hands_list = util.handDetect(candidate, subset, oriImg)
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(
                        oriImg[y : y + w, x : x + w, :]
                    ).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(
                            peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x
                        ) / float(W)
                        peaks[:, 1] = np.where(
                            peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y
                        ) / float(H)
                        hands.append(peaks.tolist())
                # Face
                faces_list = util.faceDetect(candidate, subset, oriImg)
                for x, y, w in faces_list:
                    heatmaps = self.face_estimation(oriImg[y : y + w, x : x + w, :])
                    peaks = self.face_estimation.compute_peaks_from_heatmaps(
                        heatmaps
                    ).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(
                            peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x
                        ) / float(W)
                        peaks[:, 1] = np.where(
                            peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y
                        ) / float(H)
                        faces.append(peaks.tolist())
            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            if return_is_index:
                return pose
            else:
                return draw_pose(pose, H, W)
