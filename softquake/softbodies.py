"""Module for simulating physics of softbodies following Hooke's principles."""

from __future__ import annotations
from vectors import Vector


class Softbody:
    """An object which encapsulates a set of nodes and links."""

    nodes: list[Node]
    links: list[Link]

    def __init__(self, nodes: list[Node], links: list[Link]) -> None:
        """Create a softbody from a list of nodes and list of links."""
        self.nodes = nodes
        self.links = links


class Node:
    """A point mass particle that implements Euler integration."""

    mass: float
    position: Vector
    velocity: Vector
    acceleration: Vector
    force: Vector

    def __init__(self, mass: float, position: Vector) -> None:
        """Create a node from mass and position that is static."""
        self.mass = mass
        self.position = position
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.force = Vector(0, 0)


class Link:
    """A massless Hookean spring that features a pair of spring stiffness and dampening forces."""

    nodes: tuple[Node, Node]
    length: float
    stiffness: float
    dampening: float

    def __init__(
        self,
        nodes: tuple[Node, Node],
        stiffness: float,
        dampening: float,
        length: float | None = None,
    ) -> None:
        """Create a link from a pair of nodes, stiffness coefficient, dampening coefficient and length."""
        self.nodes = nodes
        if length is None:
            self.length = Vector.dist(nodes[0].position, nodes[1].position)
        else:
            self.length = length
        self.stiffness = stiffness
        self.dampening = dampening

    def get_vector(self) -> Vector:
        """Get the displacement vector of the link."""
        return self.nodes[1].position - self.nodes[0].position

    def get_unit(self) -> Vector:
        """Get the displacement unit vector of the link."""
        vector = self.get_vector()
        return vector.div(Vector.len(vector))

    def get_length(self) -> float:
        """Get the momentary length of the link."""
        return Vector.dist(self.nodes[0].position, self.nodes[1].position)

    def get_displacement(self) -> float:
        """Get the expansion/contraction of the link from its resting configuration (positive/negative)."""
        return self.get_length() - self.length

    def get_velocity(self) -> float:
        """Get the speed of the expansion/contraction of the link (positive/negative)."""
        return (
            Vector.dot(
                self.nodes[1].velocity - self.nodes[0].velocity,
                self.nodes[1].position - self.nodes[0].position,
            )
            / self.get_length()
        )

    def get_stiffness_force(self) -> float:
        """Get the spring stiffness force expansion/contraction (positive/negative)."""
        return -self.stiffness * self.get_displacement()

    def get_dampening_force(self) -> float:
        """Get the spring dampening force expansion/contraction (positive/negative)."""
        return -self.dampening * self.get_velocity()

    def get_force(self) -> float:
        """Get the spring force expansion/contraction (positive/negative)."""
        return self.get_stiffness_force() + self.get_dampening_force()


if __name__ == "__main__":
    from time import sleep

    print(
        "This program is just a library file and is not meant to be executed directly."
    )
    sleep(5)
