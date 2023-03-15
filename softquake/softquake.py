from __future__ import annotations
from math import tau, sin
from softbodies import Softbody, Node, Link
from vectors import Vector


class Sine:
    frequency: float
    amplitude: float
    phase: float

    def get_value(self, time: float) -> float:
        return self.amplitude * sin(tau * (self.frequency * time - self.phase))


class Signal:
    sines: list[Sine]

    def get_value(self, time: float) -> float:
        return sum(sine.get_value(time) for sine in self.sines)


class Plate:
    sines: list[Sine]
    position: Vector
    velocity: Vector
    acceleration: Vector

    def __init__(self) -> None:
        pass

    def set_values(self, time: float) -> :