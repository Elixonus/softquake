from math import tau
from os import path, mkdir, rmdir, remove
from glob import glob
import cairo
import ffmpeg
import matplotlib.pyplot as plt
from softquake import RigidPlate, Sine
from softbodies import Softbody, Node, Link
from vectors import Vector

if not path.exists("snapshots"):
    mkdir("snapshots")

fps = 30
ips = 100
delta = 1 / (fps * ips)
time = 0
shot = 0

plate = RigidPlate(sines=[Sine(frequency=0.4, amplitude=1),
                          Sine(frequency=1, amplitude=0.2)], width=4, nodes=[])

nodes = []

for x in range(4):
    for y in range(8):
        node = Node(mass=1, position=Vector(3 * (x / 3 - 0.5), 6 * y / 7).add(plate.position))
        nodes.append(node)

links = []

for x in range(3):
    for y in range(8):
        link = Link(nodes=(nodes[8 * x + y], nodes[8 * (x + 1) + y]), stiffness=5000, dampening=20)
        links.append(link)

for x in range(4):
    for y in range(7):
        link = Link(nodes=(nodes[8 * x + y], nodes[8 * x + (y + 1)]), stiffness=5000, dampening=20)
        links.append(link)

for x in range(3):
    for y in range(7):
        link = Link(nodes=(nodes[8 * x + y], nodes[8 * (x + 1) + (y + 1)]), stiffness=5000, dampening=20)
        links.append(link)

for x in range(4):
    node = nodes[8 * x]
    plate.nodes.append(node)

for t in range(2):
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
        context.set_source_rgb(0.5, 1, 1)
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

        surface.write_to_png(f"snapshots/{shot:05}.png")
        shot += 1

ffmpeg.input("snapshots/%05d.png", framerate=fps).output("softquake.mp4").run(overwrite_output=True, quiet=False)

files = glob("snapshots/*")
for file in files:
    remove(file)

rmdir("snapshots")
