import cv2
import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt
import random
import sys
import datetime
import os
from sklearn.cluster import DBSCAN



data_file = 'pp-07091724.pkl'
AoV = 30/180*np.pi
cam_w = 640
cam_h = 480
xlim = [-2, 2]
ylim = [0, 4]
colors = [np.random.rand(3, ) for _ in range(20)]

def main():
    data = read_data(data_file)
    print(f'Found {len(data)} frames')
    plt.ion()
    fig = plt.figure(clear=True)
    ax0 = fig.add_subplot(111)
    cv2.namedWindow('img')
    
    i = 2000
    while i < len(data):
        print(f'{i}/{len(data)}')
        cmd = read_frame_with_display(data[i], ax0)
        if cmd == '+10':
            i += 10
        else:
            i += 1


def read_frame_with_display(data, ax0):
    boxes, img, frames = data
    if len(boxes) == 0:
        return

    ax0.cla()
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    # ax0.set_xlabel('x (m)')
    # ax0.set_ylabel('y (m)')
    plt.show()

    cv2.imshow('img', img)


    if len(frames) < 4:
        return
    # xs, ys, zs = np.split(frames[1].T, 3)
    # ax0.plot(xs, ys, 'b.')

    # xs, ys, zs = np.split(frames[3].T, 3)
    # ax0.plot(xs, ys, 'g.')

    clusters = cluster_DBSCAN(frames[1])
    if clusters is None:
        return

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = colors[i]
        cam_left = (x1 - cam_w/2)/(cam_w/2) * AoV
        cam_right = (x2 - cam_w/2)/(cam_w/2) * AoV
        for angle in [cam_left, cam_right]:
            if angle < 0:
                ax0.plot([0, xlim[0]], [
                    0, xlim[0]/np.tan(angle)], c=color)
            elif angle > 0:
                ax0.plot([0, xlim[1]], [
                    0, xlim[1]/np.tan(angle)], c=color)
            else:
                ax0.plot([0, 0], [0, ylim[1]], c=color)

            

    plt.waitforbuttonpress(timeout=0.005)
    # ls, = ax0.plot([], [], 'r.')

    # for c in clusters:
    #     cxs, cys, czs = np.split(c.T, 3)

    #     ls.set_xdata(cxs)
    #     ls.set_ydata(cys)
        # plt.waitforbuttonpress(timeout=0.005)
        # key = cv2.waitKey()
        # if key == 27:
        #     sys.exit(0)
        # if key == ord('t'):
        #     label = True
        # if key == ord('n'):
        #     return
        # if key == ord('='):
        #     return '+10'

    left_bound = -AoV
    right_bound = AoV

    for c in clusters:
        cxs, cys, czs = np.split(c.T, 3)
        centroid = np.average(c, axis=0)
        leftmost = np.min(c, axis=0)
        rightmost = np.max(c, axis=0)
        a1 = np.arctan(centroid[0]/centroid[1])
        a2 = np.arctan(leftmost[0]/leftmost[1])
        a3 = np.arctan(rightmost[0]/rightmost[1])
        ax0.plot(cxs, cys, 'b.')

        label = 'None'

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cam_left = (x1 - cam_w/2)/(cam_w/2) * AoV
            cam_right = (x2 - cam_w/2)/(cam_w/2) * AoV

            # ignore if centre out of bound
            if a1 < cam_left or a1 > cam_right:
                continue

            # ignore if centre not close to camera centre
            if not close_to(a1, (cam_left + cam_right)/2):
                continue


            # true if two overlap by 80%
            if close_to(a2, cam_left) and close_to(a3, cam_right) and (a3-a2)/(cam_right - cam_left)>0.7:
                ax0.plot(cxs, cys, 'r.')
                label = 'True'
                break

            # if not far_to(a2, cam_left) and not far_to(a3, cam_right):
            #     ax0.plot(cxs, cys, 'g.')
            #     label = 'Discard'

        # false if in aov and not fit into any camera box
        if label == 'None' and a2 > left_bound and a3 < right_bound:
            if (a3-a2)/(cam_right - cam_left) < 0.5 or far_to(a2, cam_left) or far_to(a3, cam_right):
                label = 'False'
                ax0.plot(cxs, cys, 'y.')

    plt.waitforbuttonpress()
    return 


def close_to(a, b):
    return abs(a-b) < 0.1

def far_to(a, b):
    return abs(a-b) > 0.3

def read_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def cluster_DBSCAN(data, min_points=20):
    # pdb.set_trace()
    if not data.any() or data.shape[0] < 10:
        return None
    model = DBSCAN(eps=0.06)
    model.fit((data[:, :2]))
    labels = model.labels_
    clusters = []

    for _, class_idx in enumerate(np.unique(labels)):
        if class_idx != -1:
            class_data = data[labels == class_idx]
            if class_data.shape[0] < min_points:
                continue
            clusters.append(class_data)

    return clusters


if __name__ == '__main__':
    main()


