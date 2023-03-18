from math import pi, tau, sin, cos, sqrt, atan2
from random import random
from os import path, mkdir, rmdir, remove
from glob import glob
import cairo
import ffmpeg
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from softquake import RigidPlate, Sine, Load, Sensor
from softbodies import Softbody, Node, Link
from vectors import Vector

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

sines = [Sine(frequency=0.5, amplitude=1)]

plate.sines = sines

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

nodes = []

for point in points:
    node = Node(mass=1e3, position=Vector(point[0], point[1]))
    nodes.append(node)

plate.nodes.extend([nodes[0], nodes[1], nodes[2]])
plate.set_kinematics(time)
plate.set_nodes(0.8)

loads = [Load(node=nodes[-3], force=Vector(1000, 0)),
         Load(node=nodes[-6], force=Vector(1000, 0)),
         Load(node=nodes[-9], force=Vector(1000, 0))]

sensor = Sensor(node=nodes[-2])

delaunay = Delaunay(points)
simplices = delaunay.simplices

links = []
triangles = []

for simplex in simplices:
    def add_link_maybe(n1, n2):
        for link in links:
            if ((link.nodes[0] == nodes[n1] and link.nodes[1] == nodes[n2]) or
                    (link.nodes[0] == nodes[n2] and link.nodes[1] == nodes[n1])):
                return link
        link = Link(nodes=(nodes[n1], nodes[n2]), stiffness=6e6, dampening=1e3)
        links.append(link)
        return link

    link1 = add_link_maybe(simplex[0], simplex[1])
    link2 = add_link_maybe(simplex[1], simplex[2])
    link3 = add_link_maybe(simplex[2], simplex[0])

    triangle = ([nodes[simplex[0]], nodes[simplex[1]], nodes[simplex[2]]], [link1, link2, link3])
    triangles.append(triangle)

print("Starting the simulation physics and animation loops.")

for t in range(5):
    for s in range(fps):
        for i in range(ips):
            plate.set_kinematics(time)
            plate.set_nodes(0.8)

            for node in nodes:
                node.force.set(Vector(0, -10 * node.mass))

            for load in loads:
                if load.node not in plate.nodes:
                    node.force += load.force

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

            sensor.record(time)

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
            ratio = actual_area / natural_area

            context.move_to(triangle[0][0].position.x, triangle[0][0].position.y)
            context.line_to(triangle[0][1].position.x, triangle[0][1].position.y)
            context.line_to(triangle[0][2].position.x, triangle[0][2].position.y)
            context.close_path()
            if ratio < 1:
                context.set_source_rgb(1, min(max(20 * (ratio - 1) + 1, 0), 1), min(max(20 * (ratio - 1) + 1, 0), 1))
            else:
                context.set_source_rgb(min(max(20 * (1 - ratio) + 1, 0), 1), min(max(20 * (1 - ratio) + 1, 0), 1), 1)
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

        for load in loads:
            if load.force.len() < 1e-5:
                continue

            context.save()
            context.translate(load.node.position.x, load.node.position.y)
            context.rotate(atan2(-load.force.y, -load.force.x))
            context.translate(0.3, 0)
            context.move_to(0, 0)
            context.line_to(0.3, 0.3)
            context.line_to(0.3, 0.08)
            context.line_to(0.9, 0.12)
            context.line_to(0.9, -0.12)
            context.line_to(0.3, -0.08)
            context.line_to(0.3, -0.3)
            context.close_path()
            context.set_source_rgb(1, 1, 0)
            context.fill_preserve()
            context.set_line_width(0.1)
            context.set_source_rgb(0, 0, 0)
            context.stroke()
            context.restore()

        context.save()
        context.translate(sensor.node.position.x, sensor.node.position.y)
        context.move_to(0.3, 0)
        context.line_to(0, 0)
        context.line_to(0, 0.3)
        context.line_to(0, 0)
        context.line_to(-0.3, 0)
        context.line_to(0, 0)
        context.line_to(0, -0.3)
        context.line_to(0, 0)
        context.line_to(0.3, 0)
        context.arc(0, 0, 0.3, 0, tau)
        context.set_line_width(0.2)
        context.set_source_rgb(0, 0, 0)
        context.stroke_preserve()
        context.set_line_width(0.05)
        context.set_source_rgb(1, 1, 0)
        context.stroke()
        context.restore()

        surface.write_to_png(f"images/{shot:05}.png")
        shot += 1

print("Assembling the video file using the contents of the images folder.")
ffmpeg.input("images/%05d.png", framerate=fps).output("video.mp4").run(overwrite_output=True, quiet=False)

print("Removing the images folder.")
files = glob("images/*")
for file in files:
    remove(file)
rmdir("images")

t = np.array(sensor.times)
d = np.array(sensor.positions_x)
v = np.array(sensor.velocities_x)
a = np.array(sensor.accelerations_x)

plt.style.use("dark_background")
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
fig.suptitle("Kinematics of the sensor node.")
ax1.plot(t, d, color="red")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Displacement (m)")
ax2.plot(t, v, color="dodgerblue")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity (m/s)")
ax3.plot(t, a, color="magenta")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Acceleration (m/s/s)")

fig.savefig("figure.png")
