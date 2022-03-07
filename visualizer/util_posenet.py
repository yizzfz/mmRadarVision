import random
import pickle
import numpy as np
import datetime
import cv2
import os
import scipy.io as sio
import time

np.set_printoptions(precision=4, suppress=True)

input_dim = (200, 150, 1)
output_dim = (6, 80, 60)


def vis(x):
    x = x-np.min(x)
    x = x/np.max(x)*255
    x = x.astype(np.uint8)
    x = cv2.resize(x, (x.shape[1]*4, x.shape[0]*4))
    cv2.imshow('vis', x)
    cv2.waitKey()


def plot_x(x):
    # dd = np.zeros((100, 60), dtype=np.uint8)
    # for i in range(100):
    #     for j in range(60):
    #         if np.sum(x[i, j]) > 0:
    #             dd[i, j] = 255
    return x


def plot_y(y):
    gt_sum = np.zeros(output_dim[1:], dtype=float)
    y = (y-np.min(y))/(np.max(y)-np.min(y))*255
    for i in y:
        i = i.astype(float)
        gt_sum += i
    gt_sum = np.clip(gt_sum, 0, 255)
    gt_sum = gt_sum.astype(np.uint8)
    gt_sum = cv2.resize(gt_sum, (360, 640))
    return gt_sum

# def softmax2D(x):
#     inshape = x.shape
#     x = x.reshape(inshape[0], inshape[1], -1).astype(np.float16)
#     x = softmax(x, axis=2)
#     x = x.reshape(inshape)
#     return x


'''
vs_gt = [
    0, 1,   # shoulder
    2,      # waist
    3, 4,   # knee
    5       # head
    ---
    6       # shoulder mid
]
'''
pairs = [
    (5, 6),
    (6, 0), (6, 1),
    (0, 2), (1, 2),
    (2, 3), (2, 4),
]


def mid(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return int((x1+x2)/2), int((y1+y2)/2)


'''
mode 0: origin
mode 1: beauti
'''


def plot_y_inter(y, mode=1):
    new = (output_dim[1]*8, output_dim[2]*8)
    out = np.zeros(new, dtype=np.uint8)
    pts = []
    waist = None
    for v_i, i in enumerate(y):
        idx = np.argmax(i)
        row = int(idx / output_dim[2] * 8)
        col = (idx % output_dim[2]) * 8
        rad = 3 if v_i != 5 else 12
        cv2.circle(out, (col, row), rad, 255, -1)
        pts.append((col, row))
        if v_i == 2:
            waist = col, row

    if mode == 1:
        pts.append(mid(pts[0], pts[1]))
        for v_i, i in enumerate(y):
            if v_i == 3 or v_i == 4:
                idx = np.argmax(i)
                row = int(idx / output_dim[2] * 8)
                col = (idx % output_dim[2]) * 8
                pts.append((col, row+np.abs(row-waist[1])))
        pp = pairs + [(3, 7), (4, 8)]
        for (p1, p2) in pp:
            if np.sum(pts[p1]) != 0 and np.sum(pts[p2]) != 0:
                cv2.line(out, pts[p1], pts[p2], 255)
    return out


def plot_x_y(x, y):
    h, w = y.shape
    x = x[:, :, 0]
    x = cv2.resize(x, (w, h))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = x
    out[:, :, 1] = y
    return out
