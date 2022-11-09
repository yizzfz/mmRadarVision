from config import *
from collections import deque
import cv2
import time
import struct
import os
import math

import numpy as np
import pickle
import platform
import pdb
import traceback

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from sklearn.cluster import DBSCAN
from scipy import stats
from shapely.geometry.point import Point
from shapely import affinity
from .MinimumBoundingBox import MinimumBoundingBox
from util import cluster_DBSCAN
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

# some hyper parameters
nstd = 2

class FrameBase:
    """Class to represent a scene that contains a number of objects."""
    def __init__(self):
        self.obj_list = []
        return

    def update(self):
        raise NotImplementedError

    def get_objs(self):
        """Return live objects."""
        return [obj for obj in self.obj_list if obj.live()]
    
    def get_drawings(self, ret_label=False):
        """Get drawings for each object in the scene."""
        if not ret_label:
            return [obj.get_drawing() for obj in self.obj_list if obj.live()]
        return [(obj.get_drawing(), obj.confidence) for obj in self.obj_list if obj.live()]

    def debug(self):
        print('There are %d objs' % (len(self.obj_list)))
        for c in self.obj_list:
            c.debug()

class Frame(FrameBase):
    def update(self, new_cubes):
        """Update the list of objects in the frame"""
        cens1 = np.asarray([x.get_centroid_xy() for x in new_cubes])
        cens2 = np.asarray([x.get_centroid_xy() for x in self.obj_list])
        dm = None
        if len(self.obj_list) > 0 and len(new_cubes) > 0:
            dm = distance_matrix(cens1, cens2)
        for i, cube in enumerate(new_cubes):
            matched = False
            if dm is not None:
                j = np.argmin(dm[i])
                if dm[i, j] < 1:
                    self.obj_list[j].add_cube(cube)
                    matched = True
            if not matched:
                self.obj_list.append(Obj(cube))

        for obj in self.obj_list:
            obj.dec()

        self.obj_list = [obj for obj in (self.obj_list) if obj.confidence > 0]
        return


class Obj:
    """Class to represent an object (a person)."""
    def __init__(self, cube):
        self.centroid = cube.get_centroid_xy()
        self.height = cube.height
        self.confidence = cube.confidence
        self.start_time = time.time()
        self.bases = [(cube.lenA, cube.lenB)]
        self.base_len = (0, 0)
        self.history = deque([cube], 500)
        self.effective_history_len = 15

    def debug(self):
        print(f'object confidence {self.confidence}, has lived for {len(self.history)} frames')
        # for c in self.history:
        #     print(c.get_centroid_xy())
    
    def add_cube(self, cube):
        """Add a cube to an object"""
        self.history.append(cube)
        self.update()

    def get_average_cube(self):
        """Get a cube that reprents the object"""
        cube_to_display = self.get_drawing_average()
        vec1 = cube_to_display.vec1[0:2]
        vec2 = cube_to_display.vec2[0:2]

        vec = [(math.hypot(vec1[0], vec1[1]), np.arctan2(*vec1[::-1])),
               (math.hypot(vec2[0], vec2[1]), np.arctan2(*vec2[::-1]))]
        vec.sort()

        lenA, lenB = self.base_len
        # print(vec1, vec2, np.degrees(vec[0][1]), np.degrees(vec[1][1]), lenA, lenB)

        vec1=np.asarray(
            (lenA*np.cos(vec[0][1]), lenA*np.sin(vec[0][1])))/2
        vec2=np.asarray(
            (lenB*np.cos(vec[1][1]), lenB*np.sin(vec[1][1])))/2
        
        c = self.centroid
        p1 = c+vec1+vec2
        p2 = c+vec1-vec2
        p3 = c-vec1-vec2
        p4 = c-vec1+vec2

        height = self.height
        cube = Cube((p1, p2, p3, p4), height)
        return cube

    def get_centroid_xy(self):
        return self.centroid

    def get_height(self):
        return self.height

    def get_drawing(self, color='blue'):
        """Get a drawable cube"""
        cube = self.get_average_cube()
        return cube.get_drawing(color=color)

    def get_drawing_max(self):
        """Get a drawable cube by looking for the maximum confidence"""
        lmax = 0
        cube_to_display = None
        history = list(self.history)[-self.effective_history_len:]
        for c in history:
            if c.confidence > lmax:
                cube_to_display = c
                lmax = c.confidence

        assert(cube_to_display is not None)
        return cube_to_display

    def get_drawing_first(self):
        """Get a drawable cube (the first cube)"""
        return self.history[0]

    def get_drawing_average(self):
        """Get a drawable cube that is the average of all cubes"""
        pts = []
        history = history = list(self.history)[-self.effective_history_len:]
        for c in history:
            pts += c.base
        bb = MinimumBoundingBox(pts)
        cp = bb.corner_points
        cp = list(cp)
        cp.sort()
        cp[-2:] = reversed(cp[-2:])
        avg = Cube(cp, self.height)
        return avg

    def dec(self):
        """Decrease the liveness"""
        self.confidence -= 1

    def live(self):
        return self.confidence > 10000 or len(self.history) > 125

    def update(self):
        """Update the object each frame"""
        history = list(self.history)[-self.effective_history_len:]
        cs = [c.get_centroid_xy() for c in history]
        hs = [c.height for c in history]
        weight = [c.confidence for c in history]
        x, y = np.average(np.asarray(cs), weights=weight, axis=0)
        h = np.average(np.asarray(hs), weights=weight, axis=0)
        self.centroid = x, y
        self.height = h
        self.confidence = sum([c.confidence for c in self.history])
        self.bases = [(c.lenA, c.lenB) for c in self.history]    
        self.base_len = stats.trim_mean(self.bases, 0.1)

    def get_history(self):
        return [x.get_centroid_xy() for x in self.history]



class Cube:
    """A cube that represents an object and can be drawn."""
    def __init__(self, base, height, confidence=2):
        self.base = base
        p1, p2, p3, p4 = base
        side_length = [math.hypot(p2[0]-p1[0], p2[1]-p1[1]), math.hypot(p3[0]-p2[0], p3[1]-p2[1])]
        side_length.sort()
        self.lenA = side_length[0]
        self.lenB = side_length[1]
        self.o = np.asarray((p1[0], p1[1], 0))
        self.vec1 = np.asarray((p2[0]-p1[0], p2[1]-p1[1], 0))
        self.vec2 = np.asarray((p3[0]-p2[0], p3[1]-p2[1], 0))
        self.vec3 = np.asarray((0, 0, height))
        self.height = height
        self.confidence = confidence

    def is_valid(self):
        return self.lenA < 2 and self.lenB < 2 and self.height < 3

    def get_drawing(self, color='blue', height=None):
        """Get a drawable cube"""
        if height is None:
            return draw_cube(self.o, self.vec1, self.vec2, self.vec3, color)
        else:
            vec3 = np.asarray((0, 0, height))
            return draw_cube(self.o, self.vec1, self.vec2, vec3, color)

    def close_to(self, cube_B):
        """Check if two cubes are close"""
        x1, y1 = self.get_centroid_xy()
        x2, y2 = cube_B.get_centroid_xy()
        c1 = np.sqrt((x1-x2)**2 + (y1-y2)**2) < 0.01
        area1 = self.get_area()
        area2 = self.get_area()
        c2 = (area1-area2)/area2 < 0.1
        return c1 and c2

    def get_centroid_xy(self):
        x, y, z = self.o + self.vec1/2 + self.vec2/2
        return (x, y)

    def get_bounding_box(self, color):
        return Polygon(self.base, alpha=0.5, color=color)

    def get_area(self):
        x = self.vec1[0]**2 + self.vec1[1]**2
        y = self.vec2[0]**2 + self.vec2[1]**2
        return np.sqrt(x*y)

  
class Cluster:
    """A cluster (ellipse) that represents an object"""
    def __init__(self, data):
        # input (n, 3)
        assert(data.shape[1]==3)
        self.data = reject_outliers(data)
        self.centroid = np.average(data, axis=0)
        rand = np.random.randn(*data.shape)/1e6
        data = data + rand
        self.cov = np.cov(data[:, 0:2])
        self.bound_x = (np.min(data[:, 0]), np.max(data[:, 0]))
        self.bound_y = (np.min(data[:, 1]), np.max(data[:, 1]))
        self.bound_z = (0, np.max(data[:, 2]))
        self.height = self.bound_z[1]

        self.bb = MinimumBoundingBox(data[:, :2])
        cp = self.bb.corner_points
        cp = list(cp)
        cp.sort()
        cp[-2:] = reversed(cp[-2:])
        self.corners = cp

        vals, vecs = eigsorted(self.cov)

        if np.isnan(vals).any() or np.any(vals < 0):
            self.have_ellipse = False
            return
        self.have_ellipse = True
        self.etheta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        self.ewidth, self.eheight = 2 * nstd * np.sqrt(vals)
        self.ellipse_vert = None
        self.ellipse_art = None
        

    def get_centroid(self):
        """The x-y-z centroid of the cluster"""
        return self.centroid

    def get_centroid_xy(self):
        """The x-y centroid of the cluster"""
        return self.centroid[:2]

    def get_bounding_box(self, color='blue'):
        """The minimal bounding box of the cluster"""
        return Polygon(self.corners, alpha=0.5, color=color)

    def get_ellipse_artist(self, color='black'):
        """Drawable ellipse that represnets the cluster"""
        if self.have_ellipse:
            if self.ellipse_art is None:
                self.ellipse_art = Ellipse(xy=self.centroid, width=self.ewidth,
                               height=self.eheight, angle=self.etheta, 
                               alpha=0.5, color=color)
            return self.ellipse_art
        else:
            return None

    def get_ellipse_poly(self):
        """Drawable ellipse that represnets the cluster"""
        if self.have_ellipse:
            if self.ellipse_vert is None:
                self.ellipse_vert = create_ellipse(self.centroid, self.ewidth, self.eheight, self.etheta)
            return self.ellipse_vert
        else:
            return None

    def get_bound(self):
        """Boundary of the cluster"""
        return self.bound_x, self.bound_y, self.bound_z

    def intersects(self, cluster_B):
        """Check if two clsuters overlap"""
        if not (self.have_ellipse and cluster_B.have_ellipse):
            return False

        e1 = self.get_ellipse_poly()
        e2 = cluster_B.get_ellipse_poly()

        if e1.area < 1e-5 or e2.area < 1e-5:
            return False

        inter = e1.intersection(e2)
        th = 0.2
        res = (inter.area/e1.area > th or inter.area / 
               e2.area > th or self.distance_to(cluster_B) < 0.15)
        return res

    def close_to(self, cluster_B):
        """Check if a cluster is close to another"""
        return self.distance_to(cluster_B) < 0.4

    def max_bound(self, cluster_B):
        """The boundary of the union of two clusters"""
        (xmin1, xmax1), (ymin1, ymax1), (zmin1, zmax1) = self.get_bound()
        (xmin2, xmax2), (ymin2, ymax2), (zmin2, zmax2) = cluster_B.get_bound()

        return np.asarray((min(xmin1, xmin2), max(xmax1, xmax2), 
                           min(ymin1, ymin2), max(ymax1, ymax2),
                           min(zmin1, zmin2), max(zmax1, zmax2)))

    def min_bound(self, cluster_B):
        """The boundary of the intersection of two clusters"""
        (xmin1, xmax1), (ymin1, ymax1), (zmin1, zmax1) = self.get_bound()
        (xmin2, xmax2), (ymin2, ymax2), (zmin2, zmax2) = cluster_B.get_bound()

        return np.asarray((max(xmin1, xmin2), min(xmax1, xmax2),
                           max(ymin1, ymin2), min(ymax1, ymax2),
                           min(zmin1, zmin2), max(zmax1, zmax2)))


    def distance_to(self, cluster_B):
        """The distance between the centriods of two clusters"""
        c1 = self.centroid
        c2 = cluster_B.get_centroid()
        dist = (c1-c2)**2
        dist = np.sqrt(np.sum(dist[0:2]))
        return dist

    def __len__(self):
        return self.data.shape[0]


'''
   7----8
  /|   /|
 / |  / |
5----6  |
|  3-|--4
| /  | /
|/   |/
1----2
'''

def draw_cube(origin, vec1, vec2, vec3, color='blue'):
    """Draw a cube as a collection of 3D rectangles"""
    p_12 = [origin, origin+vec1]
    p_34 = [p+vec2 for p in p_12]
    p_56 = [p+vec3 for p in p_12]
    p_78 = [p+vec3 for p in p_34]
    p_all = np.asarray(p_12+p_34+p_56+p_78)

    f_front = p_12 + p_56[::-1]
    f_back = p_34+p_78[::-1]
    f_bot = p_12+p_34[::-1]
    f_top = p_56+p_78[::-1]
    f_left = [p_all[i] for i in [0, 2, 6, 4]]
    f_right = [p_all[i] for i in [1, 3, 7, 5]]
    f_all = [f_front, f_back, f_bot, f_top, f_left, f_right]

    faces = Poly3DCollection(f_all, linewidth=0.1, edgecolors='k', alpha=0.1)
    faces.set_facecolor(color)
    # faces.set_alpha(0.1)
    return faces


# def create_cube_from_two_clusters(c1, c2):
#     bound = c1.min_bound(c2)
#     xlen, ylen, zlen = bound[1::2]-bound[0::2]
#     origin = bound[0::2]
#     vec1 = np.asarray((xlen, 0, 0))
#     vec2 = np.asarray((0, ylen, 0))
#     height = zlen
#     return Cube(origin, vec1, vec2, height)


def create_cube_from_two_clusters(c1, c2):
    """Combine two clusters into one and draw a cube"""
    pts = c1.corners + c2.corners
    bb = MinimumBoundingBox(pts)
    cp = bb.corner_points
    cp = list(cp)
    cp.sort()
    cp[-2:] = reversed(cp[-2:])
    height = max(c1.height, c2.height)
    return Cube(cp, height, len(c1)+len(c2))

def create_cube_from_cluster(c):
    cp = c.corners
    cp = list(cp)
    cp.sort()
    cp[-2:] = reversed(cp[-2:])
    return Cube(cp, c.height, len(c))


def create_ellipse(center, width, height, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, (width), (height))
    ellr = affinity.rotate(ell, angle)
    return ellr


def eigsorted(cov):
    """Find the eigenvalues and eignevectors of a matrix"""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def rotate_and_translate(xs, ys, zs, r, t):
    data = np.asarray([xs, ys, zs]).T
    data = data @ r + t
    return data.T

def cluster_xyz(xs, ys, zs, distance=0.2, min_points=1):
    """DBSCAN clustering on point cloud"""
    if not isinstance(xs, np.ndarray) or len(xs) < 2:
        return []
    frame = np.stack((xs, ys, zs), axis=-1)
    clusters = cluster_DBSCAN(frame, min_points=min_points, distance=distance)
    res = []
    if clusters is None or len(clusters) == 0:
        return []
    for class_data in clusters:
        assert(class_data.shape[1]==3)
        if class_data.shape[0] < 3:
            continue
        cc = Cluster(class_data)
        res.append(cc)
    return res


def angle_to(point1, point2):
    """Angle of arrival of point 2 to point 1"""
    x1, y1 = point1
    x2, y2 = point2
    return np.arctan((x2-x1)/(y2-y1))


def distance_to(point1, point2):
    """Euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def reject_outliers(data, m=5):
    """data (n, 3)"""
    d = np.abs(data - np.median(data, axis=0))
    mdev = np.median(d, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.nan_to_num(d/mdev, posinf=0)
    return data[(s<m).all(axis=1)]