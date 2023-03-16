from __future__ import annotations
from math import tau, sin, cos
from softbodies import Softbody, Node, Link
from vectors import Vector


class Sine:
    frequency: float
    amplitude: float
    phase: float

    def get_sine(self, time: float) -> float:
        return self.amplitude * sin(tau * (self.frequency * time - self.phase))


class RigidPlate:
    sines: list[Sine]
    width: float
    nodes: list[Node]
    position: Vector
    velocity: Vector
    acceleration: Vector

    def __init__(self, sines: list[Sine], width: float, nodes: list[Node], time: float) -> None:
        self.sines = sines
        self.width = width
        self.nodes = nodes
        self.position = Vector(0, 0)
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.set_kinematics(time)

    def set_kinematics(self, time: float) -> None:
        self.position.set(Vector(0, 0))
        self.velocity.set(Vector(0, 0))
        self.acceleration.set(Vector(0, 0))

        for sine in self.sines:
            self.position.x += sine.amplitude * sin(tau * (sine.frequency * time + sine.phase))
            self.velocity.x += tau * sine.frequency * sine.amplitude * cos(tau * (sine.frequency * time + sine.phase))
            self.acceleration.x -= (tau * sine.frequency) ** 2 * sine.amplitude * sin(tau * (sine.frequency * time + sine.phase))

    def set_nodes(self) -> None:
        for n, node in enumerate(self.nodes):
            node.position.set(self.position)
            if len(self.nodes) > 1:
                node.position.x += 0.9 * self.width * (n / (len(self.nodes) - 1) - 0.5)
            node.velocity.set(self.velocity)
            node.acceleration.set(self.acceleration)

