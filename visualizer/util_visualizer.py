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

# some hyper parameters
nstd = 2
max_confidence = 20
AoV = 45/180*np.pi

class Frame:
    """Class to represent a scene that contains a number of objects."""
    def __init__(self):
        self.obj_list = []
        return

    def update(self, new_cubes):
        """Update the list of objects in the frame"""
        for i, c in enumerate(new_cubes):
            found = False
            for j, obj in enumerate(self.obj_list):
                if obj.find_cube(c):
                    found = True
                    break

            if not found:
                self.obj_list.append(Obj(c))

        for obj in self.obj_list:
            obj.dec()

        self.obj_list = [obj for obj in (self.obj_list) if obj.live()]
        return

    def update_with_filters(self, new_cubes, filters):
        """Update the list of objects in the frame that satisfy some conditions"""
        self.update(new_cubes)
        for obj in self.obj_list:
            radar = (0, d_ver)
            centre = obj.centroid
            angle = angle_to(centre, radar)
            
            base = obj.get_average_cube().base
            leftmost = 4
            rightmost = -4
            for pts in base:
                angle_c = angle_to(pts, radar)
                leftmost = min(leftmost, angle_c)
                rightmost = max(rightmost, angle_c)
            region = rightmost - leftmost
            found = False
            
            for left, right in filters:
                if left <= angle <= right and region/(right-left) > 0.3:
                    found = True
                    break
            obj.update_label(found)

    def get_objs(self):
        """Return live objects."""
        return [obj for obj in self.obj_list if obj.confidence > 2]
    
    def get_drawings(self, ret_label=False):
        """Get drawings for each object in the scene."""
        if not ret_label:
            return [obj.get_drawing() for obj in self.obj_list if obj.confidence > 2]

        res = []
        for obj in self.obj_list:
            if obj.confidence <= 2:
                continue
            if obj.label >= 5:
                label = 'True'
            elif obj.label > 0:
                label = 'Likely'
            elif obj.label > -5:
                label = 'Unlikely'
            else:
                label = 'False'
            res.append((obj.get_drawing(), label, obj.centroid))

        return res

        # live_objs = [obj for obj in self.obj_list if obj.confidence > 2]
        # res = []
        # for obj in live_objs:
        #     centre = obj.centroid
        #     angle = angle_to(centre, (0, d_ver))

        #     found = False
        #     for left, right in filters:
        #         if angle >= left and angle <= right:
        #             if ret_label:
        #                 res.append((obj.get_drawing(color='red'), 'True'))
        #             else:
        #                 res.append(obj.get_drawing(color='red'))
        #             found = True
        #             break
        #     if not found:
        #         if ret_label:
        #             res.append((obj.get_drawing(color='blue'), 'False'))
        #         else:
        #             res.append(obj.get_drawing(color='blue'))
        # return res


    def debug(self):
        print('There are %d objs' % (len(self.obj_list)))
        for c in self.obj_list:
            c.debug()



class Obj:
    """Class to represent an object (a person). An object can have many candidate cubes."""
    def __init__(self, cube):
        self.cube_list = [cube]
        self.centroid = cube.get_centroid_xy()
        self.height = cube.height
        self.heightL = self.height
        self.confidence = cube.confidence
        self.start_time = time.time()
        self.bases = [(cube.lenA, cube.lenB)]
        self.base_len = (0, 0)
        self.mode = 'init'
        self.label = 0
        self.history = deque([], 50)

    def debug(self):
        print('object has %d candidate cubes' % len(self.cube_list))
        for c in self.cube_list:
            print(c.confidence)

    def update_label(self, found):
        """Update the liveness of the object"""
        if found:
            self.label += 1
        else:
            self.label -= 1

        self.label = max(self.label, -5)
        self.label = min(self.label, 5)

    def find_cube(self, cube):
        """Check if a cube belongs to an object"""
        for c in self.cube_list:
            if c.close_to(cube):
                c.inc(5)
                return True

        x1, y1 = self.centroid
        x2, y2 = cube.get_centroid_xy()
        if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 0.5:
            self.add_cube(cube)
            return True
        return False
    
    def add_cube(self, cube):
        """Add a cube to an object"""
        cube.inc(5)
        self.cube_list.append(cube)
        self.update()


    def get_average_cube(self):
        """Get a cube that reprents the object"""
        cube_to_display = self.get_drawing_average()
        if self.mode == 'init':
            return cube_to_display

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

        height = self.height if (self.height - self.heightL / self.heightL > 0.1) else self.heightL
        cube = Cube((p1, p2, p3, p4), height)
        return cube


    def get_centroid_xy(self):
        cube = self.get_average_cube()
        return cube.get_centroid_xy()
        

    def get_drawing(self, color=None):
        """Get a drawable cube"""
        if color is None:
            color = 'red' if self.label == 5 else 'black' if self.label == -5 else 'blue'

        cube = self.get_average_cube()
        if self.mode == 'init':
            return cube.get_drawing(height=self.height, color=color)
        return cube.get_drawing(color=color)

        
    def get_drawing_max(self):
        """Get a drawable cube by looking for the maximum confidence"""
        lmax = 0
        cube_to_display = None
        for c in self.cube_list:
            if c.confidence > lmax:
                cube_to_display = c
                lmax = c.confidence

        assert(cube_to_display is not None)
        return cube_to_display

    def get_drawing_first(self):
        """Get a drawable cube (the first cube)"""
        return self.cube_list[0]

    def get_drawing_average(self):
        """Get a drawable cube that is the average of all cubes"""
        cubes = [c for c in self.cube_list if c.confidence > 5]
        if len(cubes) < 2:
            return self.get_drawing_max()
        pts = []
        for c in cubes:
            pts += c.base
        bb = MinimumBoundingBox(pts)
        cp = bb.corner_points
        cp = list(cp)
        cp.sort()
        cp[-2:] = reversed(cp[-2:])
        avg = Cube(cp, self.height)
        return avg


    def dec(self):
        """Decrease the liveness of all cubes after one frame"""
        for c in self.cube_list:
            c.dec()
        self.cube_list = [c for c in self.cube_list if c.confidence>0]

    def live(self):
        """The object is live if at least one cube is live"""
        for c in self.cube_list:
            if c.confidence > 0:
                return True
        return False

    def update(self):
        """Update the object each frame"""
        self.cube_list = [c for c in self.cube_list if c.confidence > 0]
        cs = [c.get_centroid_xy() for c in self.cube_list]
        hs = [c.height for c in self.cube_list]
        weight = [c.confidence for c in self.cube_list]
        x, y = np.average(np.asarray(cs), weights=weight, axis=0)
        h = np.average(np.asarray(hs), weights=weight, axis=0)
        self.centroid = x, y
        self.height = h
        self.confidence = sum([c.confidence for c in self.cube_list])

        if self.mode == 'init':
            self.bases += [(c.lenA, c.lenB) for c in self.cube_list]
            if time.time()-self.start_time > 5:
                self.mode = 'run'
                print('Learned an object')
                self.heightL = self.height

        if self.mode == 'run':
            self.base_len = stats.trim_mean(self.bases, 0.1)
        self.history.append(self.centroid)

    def get_history(self):
        return self.history



class Cube:
    """A cube that represents an object and can be drawn."""
    # def __init__(self, origin, vec1, vec2, height):
    #     self.o = origin
    #     self.vec1 = vec1
    #     self.vec2 = vec2
    #     self.vec3 = np.asarray((0, 0, height))
    #     self.height = height
    #     self.confidence = 2

    def __init__(self, base, height):
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
        self.confidence = 2

    # def get_base_ratio(self):
    #     return self.xlen/self.ylen


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


    # def get_bound(self):
    #     xmin, ymin, zmin = self.o
    #     xmax = xmin + self.xlen
    #     ymax = ymin + self.ylen
    #     zmax = zmin + self.zlen
    #     return xmin, xmax, ymin, ymax, zmin, zmax

    # def update(self, cube_B):
    #     xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = self.get_bound()
    #     xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = cube_B.get_bound()

    #     new_bound = np.asarray((min(xmin1, xmin2), max(xmax1, xmax2), 
    #                        min(ymin1, ymin2), max(ymax1, ymax2),
    #                        min(zmin1, zmin2), max(zmax1, zmax2)))

    #     self.xlen, self.ylen, self.zlen = new_bound[1::2]-new_bound[0::2]
    #     self.o = new_bound[0::2]

    def inc(self, val=1):
        """Increase the liveness of the cube"""
        self.confidence += val
        self.confidence = min(self.confidence, max_confidence)

    def dec(self, val=1):
        """Decrease the liveness of the cube"""
        self.confidence -= val

    def live(self):
        return self.confidence > 0

    def get_area(self):
        x = self.vec1[0]**2 + self.vec1[1]**2
        y = self.vec2[0]**2 + self.vec2[1]**2
        return np.sqrt(x*y)

  
class Cluster:
    """A cluster (ellipse) that represents an object"""
    def __init__(self, data):
        # input (3, n)
        assert(data.shape[0]==3)
        self.data = data
        self.centroid = np.average(data, axis=1)
        rand = np.random.randn(*data.shape)/1e6
        data += rand
        self.cov = np.cov(data[0: 2, :])
        # self.cov[0, 0] += 1e-5
        # self.cov[1, 1] += 1e-5
        self.bound_x = (np.min(data[0, :]), np.max(data[0, :]))
        self.bound_y = (np.min(data[1, :]), np.max(data[1, :]))
        # self.bound_z = (np.min(data[2, :]), np.max(data[2, :]))
        self.bound_z = (0, np.max(data[2, :]))
        self.height = self.bound_z[1]


        vals, vecs = eigsorted(self.cov)

        if np.isnan(vals).any() or np.any(vals < 0):
            self.have_ellipse = False
            return None
        self.have_ellipse = True
        self.etheta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        self.ewidth, self.eheight = 2 * nstd * np.sqrt(vals)
        self.ellipse_vert = None
        self.ellipse_art = None
        self.bb = MinimumBoundingBox(data[0: 2, :].T)
        cp = self.bb.corner_points
        cp = list(cp)
        cp.sort()
        cp[-2:] = reversed(cp[-2:])
        self.corners = cp

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
        return self.distance_to(cluster_B) < 0.2

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
    return Cube(cp, height)


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

def cluster_xyz(xs, ys, zs):
    """DBSCAN clustering on point cloud"""
    if not isinstance(xs, np.ndarray) or len(xs) < 2:
        return []
    frame = np.stack((xs, ys, zs), axis=-1)
    clusters = cluster_DBSCAN(frame)
    res = []
    if clusters is None or len(clusters) == 0:
        return []
    for class_data in clusters:
        assert(class_data.shape[1]==3)
        if class_data.shape[0] < 3:
            continue
        cc = Cluster(class_data.T)
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

def in_region(cen):
    """Check if a point is within certain AoV"""
    return (-AoV < angle_to(cen, (0, d_ver)) < AoV) and cen[1] > -1.2 and -1 < cen[0] < 1
