# -*- coding: utf-8 -*-
import random
from math import sin, cos
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations

pi = 3.14159265359

# camera model class
class Camera :

    def __init__(self, f, u0, v0, th, r, _id) :
        self.id = _id

        self.p = np.array([r*cos(th), 0, r*sin(th)])

        self.R = np.array([[sin(th), 0, -cos(th)], 
                           [0, 1, 0], 
                           [-cos(th), 0, -sin(th)]])

        self.t = - np.dot(self.R, self.p)

        self.K = np.array([[f, 0, u0],
                           [0, f, v0],
                           [0, 0, 1]])
    
    def reproject(self, xyz) :
        R_t = np.array([
            [self.R[0, 0], self.R[0, 1], self.R[0, 2], self.t[0]],
            [self.R[1, 0], self.R[1, 1], self.R[1, 2], self.t[1]],
            [self.R[2, 0], self.R[2, 1], self.R[2, 2], self.t[2]],
        ])

        xyz_1 = np.array([xyz[0], xyz[1], xyz[2], np.ones(xyz[0].size)])

        local = np.dot(R_t, xyz_1)
        for i in range(local[0].size) :
            local[0, i] = local[0, i] / local[2, i]
            local[1, i] = local[1, i] / local[2, i]
            local[2, i] = local[2, i] / local[2, i]

        uv_1 = np.dot(self.K, local)

        return uv_1
        

class NoisedCamera :

    def __init__(self, Camera, t_array, n_array, phi) : # Caemra is answer model. t_array, n_array, phi are noise paramerter for transration and rotation
        C = Camera
        nx = np.array([[0, -n_array[2], n_array[1]],
                       [n_array[2], 0, -n_array[0]],
                       [-n_array[1], n_array[0], 0]])

        dR = cos(phi)*np.identity(3) + (1 - cos(phi))*np.outer(n_array, n_array) + sin(phi)*nx

        self.R = np.dot(dR, C.R)

        self.t = C.t + t_array

    def reproject(self, xyz) :
            R_t = np.array([
                [self.R[0, 0], self.R[0, 1], self.R[0, 2], self.t[0]],
                [self.R[1, 0], self.R[1, 1], self.R[1, 2], self.t[1]],
                [self.R[2, 0], self.R[2, 1], self.R[2, 2], self.t[2]],
            ])
            xyz_1 = np.array([xyz[0], xyz[1], xyz[2], np.ones(xyz[0].size)])

            local = np.dot(R_t, xyz_1)
            for i in range(local[0].size) :
                local[0, i] = local[0, i] / local[2, i]
                local[1, i] = local[1, i] / local[2, i]
                local[2, i] = local[2, i] / local[2, i]

            uv_1 = np.dot(C.K, local)

            return uv_1



def sphere(r) : # r is radius
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.ravel(np.cos(u) * np.sin(v) * r)
    z = np.ravel(np.sin(u) * np.sin(v) * r)
    y = np.ravel(np.cos(v) * r)
    xyz = np.array([x, y, z])
    xyz_noised = xyz + np.random.normal(0, 1, (3, 200)) # custom random parameter
    
    return xyz, xyz_noised

def square_() :
    xyz = np.array([[0, 0, 0, 0],
                    [50, -50, -50, 50],
                    [50, 50, -50, -50]])
    xyz_noised = np.array([[5, -6, -10, 8],
                    [10, -40, 70, 20],
                    [2, 6, 0, -30]])

    return xyz, xyz_noised

def draw(xyz) :
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    ax.scatter(xyz[0], xyz[1], xyz[2], color="g")
    plt.show()


#--------------------------main----------------------------
file_path = "../init/initial_deta2.txt" # custom output path
f = open(file_path, "w")

xyz, xyz_noised = square_()
row, col = xyz.shape
f.write("%.i\n" % (col))

xyz_T = xyz_noised.T
for p in xyz_T :
    f.write("%.8f %.8f %.8f\n" % (p[0], p[1], p[2]))

ths = [0, pi/4, pi/2]
_id = 0

f.write("%.i\n" % (len(ths)))

for th in ths :
    C = Camera(1000, 960, 540, th, 200, _id)

    # set noize paramerters
    n = np.random.normal(0, 10, (3)) # custom random parameter
    n = n / np.linalg.norm(n) 
    phi = random.gauss(0, 0.05) # custom random parameter
    t = np.random.normal(0, 1, (3)) # custom random parameter

    C_n = NoisedCamera(C, t, n, phi)

    uv_a = C.reproject(xyz)
    uv_n = C_n.reproject(xyz_noised)

    f.write("%.i\n" % (C.id))
    f.write("%.i %.i %.i\n" % (1000, 960, 540))
    f.write("%.8f %.8f %.8f\n%.8f %.8f %.8f\n%.8f %.8f %.8f\n" % 
            (C_n.R[0, 0], C_n.R[0, 1], C_n.R[0, 2], 
             C_n.R[1, 0], C_n.R[1, 1], C_n.R[1, 2],
             C_n.R[2, 0], C_n.R[2, 1], C_n.R[2, 2]))
    f.write("%.8f %.8f %.8f\n" % (C_n.t[0], C_n.t[1], C_n.t[2]))

    for i in range(uv_a.shape[1]) :
        f.write("%.8f %.8f %.8f %.8f\n" % (uv_a[0, i], uv_a[1, i], uv_n[0, i], uv_n[1, i]))

    _id = _id + 1

    # print(np.dot(C_n.R.T, C_n.R)) # check RT * R
    print(- np.dot(C_n.R.T, C_n.t))








