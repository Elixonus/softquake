from os import path, mkdir, rmdir
import cairo
import ffmpeg
import matplotlib.pyplot as plt
from softquake import RigidPlate, Sine
from softbodies import Softbody, Node, Link
from vectors import Vector

if not path.exists("snapshots"):
    mkdir("snapshots")

fps = 60
ips = 100
time = 0

plate = RigidPlate(sines=[Sine(frequency=1, amplitude=0.2)], width=1, nodes=[])

for t in range(10):
    for s in range(fps):
        for i in range(ips):
            plate.set_kinematics(time)
            plate.set_nodes()

            time += 1 / (fps * ips)

