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
shot = 0

plate = RigidPlate(sines=[Sine(frequency=1, amplitude=0.2)], width=5, nodes=[])

for t in range(10):
    for s in range(fps):
        for i in range(ips):
            plate.set_kinematics(time)
            plate.set_nodes()

            time += 1 / (fps * ips)

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


        surface.write_to_png(f"snapshots/{shot}.png")
        shot += 1
        break
    break
