from __future__ import annotations
from math import tau, sin, cos
from softbodies import Node
from vectors import Vector


class Sine:
    frequency: float
    amplitude: float
    phase: float

    def __init__(self, frequency: float, amplitude: float, phase: float = 0) -> None:
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def get_sine(self, time: float) -> float:
        return self.amplitude * sin(tau * (self.frequency * time - self.phase))


class RigidPlate:
    sines: list[Sine]
    width: float
    nodes: list[Node]
    position: Vector
    velocity: Vector
    acceleration: Vector

    def __init__(
        self, sines: list[Sine], width: float, nodes: list[Node], time: float = 0
    ) -> None:
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
            self.position.x += sine.amplitude * sin(
                tau * (sine.frequency * time + sine.phase)
            )
            self.velocity.x += (
                tau
                * sine.frequency
                * sine.amplitude
                * cos(tau * (sine.frequency * time + sine.phase))
            )
            self.acceleration.x -= (
                (tau * sine.frequency) ** 2
                * sine.amplitude
                * sin(tau * (sine.frequency * time + sine.phase))
            )

    def set_nodes(self, extent: float) -> None:
        for n, node in enumerate(self.nodes):
            node.position.set(self.position)
            if len(self.nodes) > 1:
                node.position.x += (
                    extent * self.width * (n / (len(self.nodes) - 1) - 0.5)
                )
            node.velocity.set(self.velocity)
            node.acceleration.set(self.acceleration)
            node.force.set(node.mass * node.acceleration)


class Load:
    node: Node
    force: Vector

    def __init__(self, node: Node, force: Vector) -> None:
        self.node = node
        self.force = force


class Sensor:
    node: Node
    times: list[float]
    positions_x: list[float]
    velocities_x: list[float]
    accelerations_x: list[float]

    def __init__(self, node: Node) -> None:
        self.node = node
        self.times = []
        self.positions_x = []
        self.velocities_x = []
        self.accelerations_x = []

    def record(self, time) -> None:
        self.times.append(time)
        self.positions_x.append(self.node.position.x)
        self.velocities_x.append(self.node.velocity.x)
        self.accelerations_x.append(self.node.acceleration.x)


if __name__ == "__main__":
    from time import sleep

    print(
        "This program is just a library file and is not meant to be executed directly."
    )
    sleep(5)
