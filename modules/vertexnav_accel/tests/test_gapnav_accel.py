import vertexnav_accel
import math
import numpy as np
import random


def test_vertexnav_accel_instantiation_succeeds():
    obs = []
    for ii in range(1000):
        label = np.random.rand(4)
        label /= label.sum()
        ndt = vertexnav_accel.NoisyDetectionType(label)
        nvd = vertexnav_accel.NoisyVertexDetection(angle_rad=2 * math.pi *
                                                   random.random(),
                                                   range=10 * random.random(),
                                                   detection_type=ndt,
                                                   cov_rt=np.random.rand(2, 2))

        obs.append(nvd)


def test_vertexnav_accel_get_vis_poly_succeeds():
    ll = [0.0, 0, 4, 0, 4, 0, 4, 4, 4, 4, 0, 4, 0, 4, 0, 0]
    ll = [float(e) for e in ll]
    s1 = [0.0, 0.0, 4.0, 0.0]
    s2 = [4.0, 0.0, 4.0, 4.0]
    s3 = [4.0, 4.0, 0.0, 4.0]
    s4 = [0.0, 4.0, 0.0, 0.0]
    s5 = [1.0, 1.0, 3.0, math.pi]
    s6 = [3.0, 1.0, 1.0, math.pi]
    segs = [s1, s2, s3, s4, s5, s6]

    p1 = [0.0, 0.0]
    p2 = [4.0, 0.0]
    p3 = [4.0, 4.0]
    p4 = [0.0, 4.0]
    p5 = [1.0, 1.0]
    p6 = [3.0, math.pi]
    p7 = [1.0, math.pi]
    p8 = [3.0, 1.0]
    points = [p1, p2, p3, p4, p5, p6, p7, p8]

    for _ in range(1000):
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        fw = vertexnav_accel.FastWorld(segs, points)
        print(fw.getVisPoly(0.5, 2))
        print(fw.getVisPoly(0.5, 2))
        print(fw.getVisPoly(0.5, 2))
        print(fw.getVisPoly(0.5, 2))
        print(fw.getVisPoly(0.5, 2))
        print(fw.getVisPoly(0.5, 2))
