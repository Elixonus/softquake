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


class Plate:
    sines: list[Sine]
    position: Vector
    velocity: Vector
    acceleration: Vector

    def __init__(self, sines: list[Sine], time: float) -> None:
        self.sines = sines
        self.set_kinetics(time)

    def set_kinematics(self, time: float) -> None:
        self.position.x = 0
        self.velocity.x = 0
        self.acceleration.x = 0

        for sine in self.sines:
            self.position.x += sine.amplitude * sin(tau * (sine.frequency * time + sine.phase))
            self.velocity.x += tau * sine.frequency * sine.amplitude * cos(tau * (sine.frequency * time + sine.phase))
            self.acceleration.x -= (tau * sine.frequency) ** 2 * sine.amplitude * sin(tau * (sine.frequency * time + sine.phase))
