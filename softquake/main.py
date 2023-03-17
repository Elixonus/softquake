from math import pi, tau, sin, cos, sqrt, floor, ceil
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

sines = [Sine(frequency=1, amplitude=0.5)]

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

nodes = []

for point in points:
    node = Node(mass=1, position=Vector(point[0], point[1]))
    nodes.append(node)

links = []
triangles = []

for simplex in simplices:
    def add_link_maybe(n1, n2):
        for link in links:
            if ((link.nodes[0] == nodes[n1] and link.nodes[1] == nodes[n2]) or
                    (link.nodes[0] == nodes[n2] and link.nodes[1] == nodes[n1])):
                return link
        link = Link(nodes=(nodes[n1], nodes[n2]), stiffness=5000, dampening=20)
        links.append(link)
        return link

    link1 = add_link_maybe(simplex[0], simplex[1])
    link2 = add_link_maybe(simplex[1], simplex[2])
    link3 = add_link_maybe(simplex[2], simplex[0])

    triangle = ([nodes[simplex[0]], nodes[simplex[1]], nodes[simplex[2]]], [link1, link2, link3])
    triangles.append(triangle)

for pp in pps:
    node = nodes[pp]
    plate.nodes.append(node)

print("Starting the simulation physics and animation loops.")

for t in range(1):
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

        for triangle in triangles:
            link1, link2, link3 = triangle[1][0], triangle[1][1], triangle[1][2]
            natural_semi = 0.5 * (link1.length + link2.length + link3.length)
            actual_semi = 0.5 * (link1.get_length() + link2.get_length() + link3.get_length())
            natural_area = sqrt(natural_semi *
                                (natural_semi - link1.length) *
                                (natural_semi - link2.length) *
                                (natural_semi - link3.length))
            actual_area = sqrt(actual_semi *
                               (actual_semi - link1.get_length()) *
                               (actual_semi - link2.get_length()) *
                               (actual_semi - link3.get_length()))
            diff = actual_area - natural_area

            context.move_to(triangle[0][0].position.x, triangle[0][0].position.y)
            context.line_to(triangle[0][1].position.x, triangle[0][1].position.y)
            context.line_to(triangle[0][2].position.x, triangle[0][2].position.y)
            context.close_path()
            if diff < 0:
                context.set_source_rgb(1, min(max(1 - 5 * abs(diff), 0), 1), min(max(1 - 5 * abs(diff), 0), 1))
            else:
                context.set_source_rgb(min(max(1 - 5 * diff, 0), 1), min(max(1 - 5 * diff, 0), 1), 1)
            context.fill()

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
