import cv2
import sys
import time
import sys
import os
import numpy as np
from threading import Thread
from queue import Queue, LifoQueue

class Camera_Base:
    """Only display camera images, does not interfere with radar"""
    def __init__(self, cam=0, rotate=False, detector=None):
        self.winname = 'cam'
        self.FoV_h = 28
        self.FoV_v = 28
        self.rotate = rotate
        self.detector = detector
        cv2.namedWindow(self.winname, 0)
        vc = cv2.VideoCapture(cam)

        if not vc.isOpened():  # try to get the first frame
            print('[cam] camera not found')
            return

        # vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # vc.set(cv2.CAP_PROP_FPS, 30)

        self.w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = vc.get(cv2.CAP_PROP_FPS)
        self.frameSize = self.w, self.h

        print('[cam] Height', self.h, 'Width', self.w, 'FPS', self.fps)
        self.run = True
        self.Q = Queue(maxsize=1)
        self.vc = vc
        self.out = None

        t = Thread(target=self.start, args=())
        t.daemon = True
        t.start()

    def video_info(self):
        while self.frameSize is None:
            time.sleep(0.1)
        return self.frameSize

    def start(self):
        while self.run:
            rval, frame = self.vc.read()
            if not rval:
                print('[cam] camera failed')
                return
            frame = self.process(frame)
            self.out = frame
            cv2.imshow(self.winname, frame)
            key = cv2.waitKey(1)
        cv2.destroyWindow(self.winname)
        self.vc.release()

    def update(self, info):
        """Feed radar data into the camera module"""
        if self.Q.empty():
            self.Q.put(info)

    def process(self, frame):
        return frame

    def get(self):
        """Fetch the current camera frame"""
        return self.out

    def get_detection(self):
        return self.detector.process(self.out)
    
    def stop(self):
        self.run = False
        time.sleep(0.2)
        self.t.join()
        print('[cam] stop')


class Camera_Simple(Camera_Base):
    """Overlap radar points on top of camera iamge"""
    def __init__(self, *args, height=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.empty((0, 3))
        self.height = height
        self.h1 = int(self.h/2)
        self.h2 = int(self.h/2*3)
        self.w1 = int(self.w/2)
        self.w2 = int(self.w/2*3)

    # def make_background(self):
    #     return np.zeros((self.h*2, self.w*2, 3), dtype=np.uint8)

    def process(self, frame):
        if not self.Q.empty():
            self.data = self.Q.get()
        try:
            data_stack = np.concatenate(self.data, axis=0)[:, :3]
        except ValueError as e:
            return frame
        FoV_h = self.FoV_h
        FoV_v = self.FoV_v
        FoVrh = FoV_h/180*np.pi
        FoVrv = FoV_v/180*np.pi

        for x, y, z in data_stack:
            z -= self.height
            if y==0:
                continue
            angle_h = np.arctan(x/y)
            angle_v = -np.arctan(z/y)
            r = 2
            x = int((angle_h/FoVrh + 1) * 0.5*self.w + self.w1)
            z = int((angle_v/FoVrv + 1) * 0.5*self.h + self.h1)

            color = (0, 255, 255)
            cv2.circle(frame, (x, z), r, color, -1)
        return frame

class Camera_360(Camera_Base):
    """to use a 360 degree camera"""
    def __init__(self, cam=3, rotate=False):
        # cv2.namedWindow("full")
        cv2.namedWindow('front')
        self.vc = cv2.VideoCapture(cam, cv2.CAP_DSHOW)

        if not self.vc.isOpened():  # try to get the first frame
            print('[cam] camera not found')
            return

        self.w = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vc.get(cv2.CAP_PROP_FPS)
        self.frameSize = self.w, self.h
        print('[cam] Height', self.h, 'Width', self.w, 'FPS', self.fps)

        self.out_dim = (300, 400)
        self.run = True
        self.Q = Queue(maxsize=1)
        self.out = None

        self.t = Thread(target=self.start, args=())
        self.t.daemon = True
        self.t.start()

    def video_info(self):
        return self.out_dim

    def start(self):
        print('[cam] start')
        while self.run:
            rval, frame = self.vc.read()
            if not rval:
                print('[cam] camera failed')
                return

            front = perspective(frame, 60, 80, 180, 0,
                                self.out_dim[1], self.out_dim[0])
            self.out = front
            # cv2.imshow("full", frame)
            cv2.imshow("front", front)
            key = cv2.waitKey(1)


def perspective(frame, wFOV, hFOV, THETA, PHI, height, width, RADIUS=128):
    """Helper function for converting a 360 image into a plain image. THETA is azimuth angle, PHI is elevation angle, both in degree"""
    equ_h = frame.shape[0]
    equ_w = frame.shape[1]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * \
        np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * \
        np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y)
                     * h_interval, [width, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([height, width, 3], np.float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.zeros([height * width], np.float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(np.bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

    lon[idx1] = theta[idx1]
    lon[idx3] = theta[idx3] + np.pi
    lon[idx4] = theta[idx4] - np.pi

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy
    #for x in range(width):
    #    for y in range(height):
    #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
    #return self._img

    persp = cv2.remap(frame, lon.astype(np.float32), lat.astype(
        np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return persp


if __name__ == '__main__':
    cam = Camera_Base(0)
    cam.start()