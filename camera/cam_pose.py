import cv2
import sys
import time
import sys
import os
import numpy as np
from threading import Thread
from queue import Queue, LifoQueue
from .cam import Camera_Base, Camera_Simple

POSE_PATH = 'c:/Users/hc13414/OneDrive - University of Bristol/cam/pose-cv/AlphaPose_torch'
if not os.path.exists(POSE_PATH):
    raise ValueError(f'Path not exist: {POSE_PATH}')
sys.path.insert(1, POSE_PATH)
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt
from dataloader_webcam import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from fn import vis_frame_fast as vis_frame

class Camera_Pose(Camera_Simple):
    def __init__(self, cam=3, is360=True, pose_vis=True, rotate=False):
        self.winname = 'cam'
        self.vis = pose_vis
        self.cam = cam
        self.is360 = is360

        self.FoV_h = 30
        self.FoV_v = 40

        if self.vis:
            cv2.namedWindow(self.winname, 0)

        self.run = True
        self.Q = Queue(maxsize=1)
        self.rotate = rotate
        self.res = None
        self.frameSize = None

        self.t = Thread(target=self.start, args=())
        self.t.daemon = True
        self.t.start()

    def make_background(self):
        self.h1 = 0
        self.h2 = int(self.h)
        self.w1 = 0
        self.w2 = int(self.w)
        return np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def start(self):
        os.chdir(POSE_PATH)
        data_loader = WebcamLoader(self.cam, is360=self.is360).start()
        (_, fps, frameSize) = data_loader.videoinfo()
        self.frameSize = frameSize
        if self.rotate:
            self.w = frameSize[1]
            self.h = frameSize[0]
        else:
            self.w = frameSize[0]
            self.h = frameSize[1]
        self.fps = fps
        print('[cam] ', frameSize, 'FPS', self.fps)

        # Load detection loader
        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader(data_loader, batchSize=1).start()
        det_processor = DetectionProcessor(det_loader, queueSize=1).start()

        # Load pose model
        pose_dataset = Mscoco()
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        pose_model.cuda()
        pose_model.eval()

        sys.stdout.flush()
        batchSize = 80
        writer = DataWriter(False, None,
                            cv2.VideoWriter_fourcc(*'XVID'),
                            fps, frameSize, queueSize=1).start()

        data = None
        back = self.make_background()

        print('[cam] start')
        while self.run:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                frame = orig_img.copy()

                if not(boxes is None or boxes.nelement() == 0):
                    # Pose Estimation
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    hm = hm.cpu().data

                    writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                    res = writer.results()
                    if len(res) > 0:
                        res = res[-1]
                        frame = vis_frame(orig_img, res)
                        self.res = res
                else:
                    writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                    self.res = None

                if not self.vis:
                    continue

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                if not self.Q.empty():
                    data = self.Q.get()

                if data:
                    newframe = back.copy()
                    newframe[self.h1:self.h2, self.w1:self.w2] = frame
                    frame = self.process(newframe, data)
                cv2.imshow(self.winname, frame)
                key = cv2.waitKey(1)

        while(writer.running()):
            pass
        writer.stop()

    def get(self):
        if self.res is None:
            return None
        res = self.res['result']
        if (len(res) == 0):
            return None
        idx = np.argmax([obj['proposal_score'] for obj in res])
        kps = res[idx]['keypoints']
        kps = torch.cat((kps,
                         torch.unsqueeze((kps[5, :]+kps[6, :])/2, 0),
                         torch.unsqueeze((kps[11, :]+kps[12, :])/2, 0)))
        return np.array(kps)
