from math import pi, tau, sin, cos, sqrt
from random import random
from os import path, mkdir, rmdir, remove
from glob import glob
import cairo
import ffmpeg
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from softquake import RigidPlate, Sine
from softbodies import Softbody, Node, Link
from vectors import Vector

if path.isfile("video.mp4"):
    print("Removing the video file.")
    remove("video.mp4")

if not path.exists("images"):
    print("Making the images folder.")
    mkdir("images")
elif path.isdir("images"):
    print("Removing the contents of the images folder.")
    files = glob("images/*")
    for file in files:
        remove(file)

fps = 60
ips = 100
delta = 1 / (fps * ips)
time = 0
shot = 0

plate = RigidPlate(sines=[], width=4, nodes=[])

sines = []

for s in range(0, 100):
    sine = Sine(frequency=0.02 * s + 0.2, amplitude=5 * random() / 100, phase=random())
    sines.append(sine)

plate.sines = sines

# points = np.empty(shape=(len(nodes), 2))
points = np.array([
    [-1, 0],
    [0, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [-1, 2],
    [0, 2],
    [1, 2],
    [-1, 3],
    [0, 3],
    [1, 3],
    [-1, 4],
    [0, 4],
    [1, 4],
    [-1, 5],
    [0, 5],
    [1, 5],
    [-1, 6],
    [0, 6],
    [1, 6],
    [-1, 7],
    [0, 7],
    [1, 7],
])

pps = np.array([0, 1, 2])

delaunay = Delaunay(points)
simplices = delaunay.simplices
used = np.full(shape=(len(points), len(points)), fill_value=False, dtype=bool)

nodes = []

for point in points:
    node = Node(mass=1, position=Vector(point[0], point[1]))
    nodes.append(node)

links = []

for simplex in simplices:
    def add_link_maybe(vertex1, vertex2):
        if not used[vertex1][vertex2]:
            used[vertex1][vertex2] = True
            used[vertex2][vertex1] = True
            link = Link(nodes=(nodes[vertex1], nodes[vertex2]), stiffness=5000, dampening=20)
            links.append(link)
            return link
        return None
    add_link_maybe(simplex[0], simplex[1])
    add_link_maybe(simplex[1], simplex[2])
    add_link_maybe(simplex[2], simplex[0])

for pp in pps:
    node = nodes[pp]
    plate.nodes.append(node)

print("Starting the simulation physics and animation loops.")

for t in range(10):
    for s in range(fps):
        for i in range(ips):
            plate.set_kinematics(time)
            plate.set_nodes(0.8)

            for node in nodes:
                node.force.set(Vector(0, -9.8 * node.mass))

            for link in links:
                force = link.get_force()
                unit = link.get_unit()
                link.nodes[0].force -= unit * force
                link.nodes[1].force += unit * force

            for node in nodes:
                if node not in plate.nodes:
                    acceleration = node.acceleration.copy()
                    node.acceleration = node.force / node.mass
                    node.position += node.velocity * delta + 0.5 * node.acceleration * delta ** 2
                    node.velocity += 0.5 * (acceleration + node.acceleration) * delta

            time += delta

        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1000, 1000)
        context = cairo.Context(surface)

        context.translate(0, 500)
        context.scale(1, -1)
        context.translate(0, -500)

        context.rectangle(0, 0, 1000, 1000)
        context.set_source_rgb(0, 1, 1)
        context.fill()

        context.rectangle(0, 0, 1000, 200)
        context.set_source_rgb(0, 1, 0)
        context.fill()

        context.scale(100, 100)
        context.translate(5, 5)
        context.translate(0, -4)

        context.rectangle(plate.position.x - 0.5 * plate.width, plate.position.y - 0.5, plate.width, 0.5)
        context.set_source_rgb(0.5, 0.25, 0.125)
        context.fill_preserve()
        context.set_line_width(0.1)
        context.set_source_rgb(0, 0, 0)
        context.stroke()

        for link in links:
            context.move_to(link.nodes[0].position.x, link.nodes[0].position.y)
            context.line_to(link.nodes[1].position.x, link.nodes[1].position.y)
            context.set_line_width(0.1)
            context.set_source_rgb(0, 0, 0)
            context.stroke()

        for node in nodes:
            context.arc(node.position.x, node.position.y, 0.15, 0, tau)
            context.set_source_rgb(1, 1, 1)
            context.fill_preserve()
            context.set_line_width(0.1)
            context.set_source_rgb(0, 0, 0)
            context.stroke()

        surface.write_to_png(f"images/{shot:05}.png")
        shot += 1

print("Assembling the video file using the contents of the images folder.")
ffmpeg.input("images/%05d.png", framerate=fps).output("video.mp4").run(overwrite_output=True, quiet=False)

print("Removing the images folder.")
files = glob("images/*")
for file in files:
    remove(file)
rmdir("images")
