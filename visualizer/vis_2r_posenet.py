"""Code in this file are still under development"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import datetime
import pickle
from .util_visualizer import *
import sys
from scipy import stats
from .vis_2r_pose import Visualizer_TwoR_Vertical
from scipy import stats
from .util_posenet import plot_y_inter, plot_y


pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (18, 11), (18, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16),
    (17, 18)
]
vs = [
    5, 6, 17, # shoulder
    11, 12, 18, # waist
    7, 8, # elbow
    13, 14, # knee
    15, 16, # feet
    9, 10, # hand
]

vs_gt = [
    5, 6,  # shoulder
    18,  # waist
    7, 8,  # elbow
    13, 14,  # knee
    15, 16,  # feet
    9, 10,  # hand
]

class Visualizer_TwoR_Pose(Visualizer_TwoR_Vertical):
    """Posture data capturing using two radars"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cv2.namedWindow('1', 0)
        # cv2.namedWindow('2', 0)
        self.cam_size = None
        self.bg = None

    # plot the same way as alphapose
    def plot_pose_v1(self, pose_data, distance):
        frame = self.bg.copy()
        if pose_data is not None:

            for x, y in pose_data:
                cv2.circle(frame, (x, y), 5, 255, -1)

            for i1, i2 in pairs:
                x1, y1 = pose_data[i1]
                x2, y2 = pose_data[i2]
                cv2.line(frame, (x1, y1), (x2, y2), 255)

            frame = cv2.resize(frame, (270, 480))
            nz = np.nonzero(frame)
            left = nz[1].min()
            right = nz[1].max()
            top = nz[0].min()
            down = nz[0].max()
            frame = frame[top:down, left:right]


            ratio = distance/1.2
            height, width = frame.shape
            height = int(height*ratio)
            width = int(width*ratio)
            frame = cv2.resize(frame, (width, height))

            height, width = frame.shape
            bh = max(0, int((270-width)/2))
            bv = max(0, int((480-height)))
            frame = cv2.copyMakeBorder(frame, bv, 0, bh, bh, cv2.BORDER_CONSTANT, value=0)
            frame = cv2.resize(frame, (270, 480))

        cv2.imshow('1', frame)

    # plot all joints on one graph, return boundary
    def plot_pose_v2(self, pose_data, distance):
        frame = self.bg.copy()
        bound = None
        if pose_data is not None:

            for idx in vs:
                x, y = pose_data[idx]
                cv2.circle(frame, (x, y), 20, 255, -1)

            head = np.asarray(pose_data[0:5])
            head = np.median(head, axis=0)
            hx, hy = head.astype(int)
            # head = (int((np.min(head[:, 0])+np.max(head[:, 0]))/2),
            #         int((np.min(head[:, 1])+np.max(head[:, 1]))/2))
            cv2.circle(frame, (hx, hy), 30, 255, -1)

            # for i1, i2 in pairs:
            #     x1, y1 = pose_data[i1]
            #     x2, y2 = pose_data[i2]
            #     cv2.line(frame, (x1, y1), (x2, y2), 255)

            frame = cv2.resize(frame, (270, 480))
            nz = np.nonzero(frame)
            left = nz[1].min()
            right = nz[1].max()
            top = nz[0].min()
            down = nz[0].max()
            bound = top, down, left, right
            frame = frame[top:down, left:right]

            ratio = distance/1.2
            height, width = frame.shape
            height = int(height*ratio)
            width = int(width*ratio)
            frame = cv2.resize(frame, (width, height))

            height, width = frame.shape
            bh = max(0, int((270-width)/2))
            bv = max(0, int((480-height)/2))
            frame = cv2.copyMakeBorder(
                frame, bv, bv, bh, bh, cv2.BORDER_CONSTANT, value=0)
            frame = cv2.resize(frame, (45, 80))
            frame = cv2.GaussianBlur(frame, (7, 7), 0)
            cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX)
        cv2.imshow('1', frame)
        return bound

    # plot individual joint on seperate graphs, based on boundary
    # return 12x80x45
    def generate_gt(self, pose_data, distance, bound):
        gt = None
        if pose_data is not None:
            top, down, left, right = bound
            gt = np.zeros((12, 80, 45), dtype=np.uint8)
            frames = np.zeros((12, 1080, 1920), dtype=np.uint8)
            for i in range(len(vs_gt)):
                idx = vs_gt[i]
                x, y = pose_data[idx]
                cv2.circle(frames[i], (x, y), 50, 255, -1)

            head = np.asarray(pose_data[0:5])
            head = np.median(head, axis=0)
            hx, hy = head.astype(int)
            # head = (int((np.min(head[:, 0])+np.max(head[:, 0]))/2),
            #         int((np.min(head[:, 1])+np.max(head[:, 1]))/2))
            cv2.circle(frames[-1], (hx, hy), 80, 255, -1)

            for i in range(12):
                frame = frames[i].copy()
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.resize(frame, (270, 480))
                nz = np.nonzero(frame)
                if len(nz[0]) == 0 or len(nz[1]) == 0:
                    print('error 1 at', i)
                    cv2.imshow('2', frame)
                    return None

                frame = frame[top:down, left:right]
                if down - top <= 1 or right - left <= 1:
                    print('error 2 at', i)
                    cv2.imshow('2', frame)
                    return None

                ratio = distance/1.2
                height, width = frame.shape
                height = int(height*ratio)
                width = int(width*ratio)
                frame = cv2.resize(frame, (width, height))

                height, width = frame.shape
                bh = max(0, int((270-width)/2))
                bv = max(0, int((480-height)/2))
                frame = cv2.copyMakeBorder(
                    frame, bv, bv, bh, bh, cv2.BORDER_CONSTANT, value=0)
                frame = cv2.resize(frame, (45, 80))
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
                cv2.normalize(frame, frame, 0, 255, norm_type=cv2.NORM_MINMAX)
                gt[i] = frame.copy()

            tmp = np.concatenate((
                np.concatenate(gt[0:3], axis=0),
                np.concatenate(gt[3:6], axis=0),
                np.concatenate(gt[6:9], axis=0),
                np.concatenate(gt[9:12], axis=0)),
                axis=1
            )
            cv2.imshow('2', tmp)
        return gt

    def plot_combined(self, frame, runflag):
        if self.cam:
            if self.cam_size is None:
                self.cam_size = self.cam.video_info()
                self.bg = np.zeros(self.cam_size, dtype=np.uint8)
                print(self.bg.shape)
            pose_data = self.cam.get()
            # _, ys1, _ = self.ps1
            # _, ys2, _ = self.ps2
            # ys = np.append(ys1, ys2)
            # distance = np.median(ys) if ys.size > 0 else 1
            # self.plot_pose_v1(pose_data, distance)
            # gt = self.generate_gt(pose_data, distance, bound)
            if pose_data is not None:
                data = {'ps1': self.ps1, 'ps2': self.ps2, 'pose': pose_data}
                if self.logger:
                    self.logger.update(data, datatype='misc')

        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed is None:
            return
        if keyPressed:
            runflag.value = 0

    def finish(self):
        if self.logger:
            header = {}
            header['data_type'] = ['ps1', 'ps2', 'pose']
            header['cam_size'] = self.cam_size
            header['height'] = [self.height1, self.height2]
            self.logger.set_header(header)
        if self.cam:
            self.cam.stop()
        print('visualizer finishing')
        super().finish()


out_dim = np.asarray([100, 60, 60], dtype=np.uint8)
step = 0.02
pairs = [
    (11, 12),
    (12, 0), (12, 1),
    (0, 2), (1, 2),
    (0, 3), (1, 4),
    (2, 5), (2, 6),
    (7, 5), (6, 8),
    (3, 9), (4, 10)
]

class Visualizer_TwoR_Posenet_v0(Visualizer_TwoR_Vertical):
    """Applying a pre-trained Posenet"""
    def __init__(self, queues, nn, **kwargs):
        super().__init__(queues, **kwargs)
        self.win = 'pose'
        self.nn = nn
        cv2.namedWindow(self.win, 0)

    def ps_to_mat(self, obj, cen):
        obj = obj - cen
        obj = np.asarray(obj/step, dtype=int)
        mat = np.zeros(out_dim, dtype=np.uint8)
        cnt = 0
        for p in obj:
            p = p + out_dim/2
            p = p.astype(int)
            if p.any() < 0:
                cnt += 1
                continue
            try:
                mat[tuple(p)] += 1
            except IndexError:
                cnt += 1
                continue
        return mat

    # take two [h, w, d], make [1, h, w, d, 2]
    def frame_to_mat(self):
        xs1, ys1, zs1 = self.ps1
        xs2, ys2, zs2 = self.ps2
        zs1 = -zs1
        zs2 = -zs2
        xs = np.append(xs1, xs2)
        ys = np.append(ys1, ys2)
        zs = np.append(zs1, zs2)
        frame1 = np.asarray((zs1, xs1, ys1)).T
        frame2 = np.asarray((zs2, xs2, ys2)).T
        frame = np.asarray((zs, xs, ys)).T
        cen = stats.trim_mean(frame, 0.2)

        out1 = self.ps_to_mat(frame1, cen)
        out2 = self.ps_to_mat(frame2, cen)
        out = np.stack((out1, out2), axis=-1)
        out = np.array([out])
        return out

    def plot_y(self, y):
        old = (80, 45)
        new = (640, 360)
        out = np.zeros(new, dtype=np.uint8)
        pts = []
        for i in y:
            idx = np.argmax(i)
            row = int(idx / 45 * 8)
            col = (idx % 45) * 8
            cv2.circle(out, (col, row), 5, 255, -1)
            pts.append((col, row))
        for (p1, p2) in pairs:
            cv2.line(out, pts[p1], pts[p2], 255, 2)
        cv2.imshow(self.win, out)
        cv2.waitKey(1)

    def plot_inter(self, runflag):
        data = self.frame_to_mat()
        pred = self.nn.model_predict(data)[0]
        self.plot_y(pred)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed is None:
            return
        if keyPressed:
            runflag.value = 0


# x_output_dim_3d = np.asarray([100, 60, 60], dtype=np.uint8)
# x_output_dim = np.asarray([200, 150], dtype=np.uint8)
# y_input_dim = np.asarray([12, 200, 150], dtype=np.uint8)
# y_output_dim = np.asarray([12, 80, 60], dtype=np.uint8)
# ref_v = x_output_dim[0]/2/np.tan(np.radians(40))
# ref_h = x_output_dim[1]/2/np.tan(np.radians(30))

class Visualizer_TwoR_Posenet(Visualizer_TwoR_Vertical):
    """Applying a pre-trained Posenet"""
    def __init__(self, queues, nn, **kwargs):
        super().__init__(queues, **kwargs)
        self.win = 'pose'
        self.nn = nn
        cv2.namedWindow(self.win, 0)

    def generate_x(self, ps1, ps2):
        xs1, ys1, zs1 = ps1
        xs2, ys2, zs2 = ps2
        # zs1 = -zs1
        # zs2 = -zs2
        xs = np.append(xs1, xs2)
        ys = np.append(ys1, ys2)
        zs = np.append(zs1, zs2)
        ps = np.array((xs, ys, zs)).T
        ps_2d = []
        frame = np.zeros((x_output_dim[0], x_output_dim[1]), dtype=np.uint8)
        for i, (x, y, z) in enumerate(ps):
            z -= 1
            z = z/y*ref_v
            x = x/y*ref_h
            z = int(x_output_dim[0]/2 - z)
            x = int(x_output_dim[1]/2 + x)
            ps[i] = x, 0, z
        ps = ps[:, (0, 2)]
        # mask = np.abs(ps - np.mean(ps, axis=0)) < 3 * np.std(ps, axis=0)
        # mask = mask[:, 0] & mask[:, 1]
        # ps = ps[mask]
        ps = ps.astype(int)

        for x, z in ps:
            cv2.circle(frame, (x, z), 1, 255)
        frame = cv2.GaussianBlur(frame, (3, 3), 1)
        return frame

    def mid(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        return int((x1+x2)/2), int((y1+y2)/2)

    def plot_y_inter(self, y):
        old = (y_output_dim[1], y_output_dim[2])
        new = (y_output_dim[1]*8, y_output_dim[2]*8)
        out = np.zeros(new, dtype=np.uint8)
        pts = []
        for i in y:
            idx = np.argmax(i)
            row = int(idx / y_output_dim[2] * 8)
            col = (idx % y_output_dim[2]) * 8
            cv2.circle(out, (col, row), 3, 255, -1)
            pts.append((col, row))
        pts.append(self.mid(pts[0], pts[1]))
        for (p1, p2) in pairs:
            if np.sum(pts[p1]) != 0 and np.sum(pts[p2]) != 0:
                cv2.line(out, pts[p1], pts[p2], 255)
        return out

    def plot_inter(self, runflag):
        x = self.generate_x(self.ps1, self.ps2)
        pred = self.nn.model_predict_one(x)
        out = self.plot_y_inter(pred)
        cv2.imshow(self.win, out)
        cv2.waitKey(1)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed is None:
            return
        if keyPressed:
            runflag.value = 0


class Visualizer_TwoR_Posenet_vs6(Visualizer_TwoR_Vertical):
    def __init__(self, queues, nn, **kwargs):
        super().__init__(queues, **kwargs)
        self.win = 'pose'
        self.nn = nn
        self.x_dim = (200, 150, 1)
        self.y_dim = (6, 80, 60)
        self.ref_v = self.x_dim[0]/2/np.tan(np.radians(40))
        self.ref_h = self.x_dim[1]/2/np.tan(np.radians(30))
        cv2.namedWindow(self.win, 0)

    def generate_x(self, ps1, ps2):
        xs1, ys1, zs1 = ps1
        xs2, ys2, zs2 = ps2
        # zs1 = -zs1
        # zs2 = -zs2
        xs = np.append(xs1, xs2)
        ys = np.append(ys1, ys2)
        zs = np.append(zs1, zs2)
        ps = np.array((xs, ys, zs)).T
        ps_2d = []
        frame = np.zeros((self.x_dim[0], self.x_dim[1]), dtype=np.uint8)
        for i, (x, y, z) in enumerate(ps):
            z -= 1
            z = z/y*self.ref_v
            x = x/y*self.ref_h
            z = int(self.x_dim[0]/2 - z)
            x = int(self.x_dim[1]/2 + x)
            ps[i] = x, 0, z
        ps = ps[:, (0, 2)]
        # mask = np.abs(ps - np.mean(ps, axis=0)) < 3 * np.std(ps, axis=0)
        # mask = mask[:, 0] & mask[:, 1]
        # ps = ps[mask]
        ps = ps.astype(int)

        for x, z in ps:
            cv2.circle(frame, (x, z), 1, 255)
        frame = cv2.GaussianBlur(frame, (3, 3), 1)
        return frame

    def plot_inter(self, runflag):
        x = self.generate_x(self.ps1, self.ps2)
        pred = self.nn.model_predict(np.expand_dims(x, axis=(0, -1)))[0]
        out = plot_y(pred)
        cv2.imshow(self.win, out)
        cv2.waitKey(1)
        keyPressed = plt.waitforbuttonpress(timeout=0.005)
        if keyPressed is None:
            return
        if keyPressed:
            runflag.value = 0
