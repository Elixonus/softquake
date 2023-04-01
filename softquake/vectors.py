"""Module for dealing with two-dimensional vectors."""

from __future__ import annotations
from math import hypot
from collections.abc import Iterator


class Vector:
    """Two-dimensional vector represented in cartesian coordinates."""

    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        """Instantiate a vector from cartesian coordinates."""
        self.x = x
        self.y = y

    def __iter__(self) -> Iterator[float]:
        """Iterate through the pair of coordinates."""
        yield self.x
        yield self.y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, vector: Vector, /) -> Vector:
        """(+) Add caller vector with parameter vector, without modification to vector."""
        return self.copy().add(vector)

    def __iadd__(self, vector: Vector, /) -> Vector:
        """(+=) Add caller vector with parameter vector, with modification to caller vector."""
        return self.add(vector)

    def __sub__(self, vector: Vector, /) -> Vector:
        """(-) Subtract parameter vector from caller vector, without modification to vector."""
        return self.copy().sub(vector)

    def __isub__(self, vector: Vector, /) -> Vector:
        """(-=) Subtract parameter vector from caller vector, with modification to caller vector."""
        return self.sub(vector)

    def __mul__(self, multiplier: float, /) -> Vector:
        """(*) Multiply vector coordinates by a number, without modification to vector."""
        return self.copy().mul(multiplier)

    def __rmul__(self, multiplier: float, /) -> Vector:
        """(*) Multiply vector coordinates by a number, without modification to vector."""
        return self.copy().mul(multiplier)

    def __imul__(self, multiplier: float, /) -> Vector:
        """(*=) Multiply vector coordinates by a number, with modification to vector."""
        return self.mul(multiplier)

    def __truediv__(self, divisor: float, /) -> Vector:
        """(/) Divide vector coordinates by a number, without modification to vector."""
        return self.copy().div(divisor)

    def __itruediv__(self, divisor: float, /) -> Vector:
        """(/=) Divide vector coordinates by a number, with modification to vector."""
        return self.div(divisor)

    def __pos__(self) -> Vector:
        """(+) Return the same vector instance."""
        return self

    def __neg__(self) -> Vector:
        """(-) Flip the sign of vector coordinates, without modification to vector."""
        return self.copy().mul(-1)

    def __matmul__(self, vector: Vector, /) -> float:
        """(@) Find the dot product of two vectors."""
        return self.dot(vector)

    def __mod__(self, vector: Vector, /) -> float:
        """(%) Find the cross product of two vectors."""
        return self.cross(vector)

    def set(self, vector: Vector, /) -> Vector:
        """Set coordinates of caller vector to match parameter vector."""
        self.x = vector.x
        self.y = vector.y
        return self

    def add(self, vector: Vector, /) -> Vector:
        """Add caller vector with parameter vector, with modification to caller vector."""
        self.x += vector.x
        self.y += vector.y
        return self

    def sub(self, vector: Vector, /) -> Vector:
        """Subtract parameter vector from caller vector, with modification to caller vector."""
        self.x -= vector.x
        self.y -= vector.y
        return self

    def mul(self, multiplier: float, /) -> Vector:
        """Multiply vector coordinates by a number, with modification to vector."""
        self.x *= multiplier
        self.y *= multiplier
        return self

    def div(self, divisor: float, /) -> Vector:
        """Divide vector coordinates by a number, with modification to vector."""
        self.x /= divisor
        self.y /= divisor
        return self

    def len(self) -> float:
        """Find the distance to the origin."""
        return hypot(self.x, self.y)

    def dist(self, vector: Vector, /) -> float:
        """Find the distance between two vectors."""
        return hypot(self.x - vector.x, self.y - vector.y)

    def dot(self, vector: Vector, /) -> float:
        """Find the dot product of two vectors."""
        return self.x * vector.x + self.y * vector.y

    def cross(self, vector: Vector, /) -> float:
        """Find the cross product of two vectors."""
        return self.x * vector.y - self.y * vector.x

    def copy(self) -> Vector:
        """Copy the vector instance."""
        return Vector(self.x, self.y)


if __name__ == "__main__":
    from time import sleep

    print(
        "This program is just a library file and is not meant to be executed directly."
    )
    sleep(5)
