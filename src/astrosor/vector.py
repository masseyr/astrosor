"""
astrosor.vector — 3-element Euclidean vector
=============================================

A lightweight, mutable 3-D vector type with operator overloading,
rotation helpers, and seamless NumPy interoperability.

Construction
------------
>>> v = Vector3(1, 2, 3)          # three scalars
>>> v = Vector3([1, 2, 3])        # list / tuple
>>> v = Vector3(np.array([1,2,3]))# ndarray

Arithmetic
----------
All standard operators work element-wise between Vector3 objects or
between a Vector3 and a scalar (for multiplication/division).

    v + w, v - w, -v, 2*v, v/2, v @ w  (dot product)
"""

import numpy as np
from numpy.typing import NDArray


class Vector3:
    """Mutable 3-element Euclidean vector."""

    __slots__ = ("_v",)

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, x, y=None, z=None):
        if y is None:
            arr = np.asarray(x, dtype=np.float64).ravel()
            if arr.size != 3:
                raise ValueError(f"Expected 3 elements, got {arr.size}")
            self._v = arr.copy()
        else:
            self._v = np.array([float(x), float(y), float(z)])

    @classmethod
    def zero(cls) -> "Vector3":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def i(cls) -> "Vector3":
        """Unit vector along X."""
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def j(cls) -> "Vector3":
        """Unit vector along Y."""
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def k(cls) -> "Vector3":
        """Unit vector along Z."""
        return cls(0.0, 0.0, 1.0)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def x(self) -> float:
        return float(self._v[0])

    @property
    def y(self) -> float:
        return float(self._v[1])

    @property
    def z(self) -> float:
        return float(self._v[2])

    @x.setter
    def x(self, val: float): self._v[0] = val

    @y.setter
    def y(self, val: float): self._v[1] = val

    @z.setter
    def z(self, val: float): self._v[2] = val

    @property
    def norm(self) -> float:
        """Euclidean magnitude (same as ``abs(v)``)."""
        return float(np.linalg.norm(self._v))

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Vector3({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

    def __str__(self) -> str:
        return f"[{self.x:.6g}, {self.y:.6g}, {self.z:.6g}]"

    # ── Sequence / array protocol ─────────────────────────────────────────────

    def __len__(self) -> int:
        return 3

    def __iter__(self):
        return iter(self._v)

    def __getitem__(self, idx):
        return self._v[idx]

    def __setitem__(self, idx, val):
        self._v[idx] = val

    def __array__(self, dtype=None) -> NDArray:
        """Allow numpy to consume a Vector3 directly."""
        return self._v.astype(dtype) if dtype is not None else self._v.copy()

    # ── Equality ──────────────────────────────────────────────────────────────

    def __eq__(self, other) -> bool:
        if isinstance(other, Vector3):
            return bool(np.allclose(self._v, other._v))
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self._v))

    # ── Internal coercion ─────────────────────────────────────────────────────

    @staticmethod
    def _coerce(other) -> NDArray:
        if isinstance(other, Vector3):
            return other._v
        arr = np.asarray(other, dtype=np.float64).ravel()
        if arr.size == 3:
            return arr
        raise TypeError(
            f"Expected Vector3 or length-3 array-like, got {type(other).__name__}"
        )

    # ── Arithmetic operators ──────────────────────────────────────────────────

    def __pos__(self) -> "Vector3":
        return Vector3(self._v.copy())

    def __neg__(self) -> "Vector3":
        return Vector3(-self._v)

    def __abs__(self) -> float:
        return self.norm

    def __add__(self, other) -> "Vector3":
        return Vector3(self._v + self._coerce(other))

    def __radd__(self, other) -> "Vector3":
        if other == 0:          # enables sum([v1, v2, ...])
            return Vector3(self._v.copy())
        return Vector3(self._coerce(other) + self._v)

    def __iadd__(self, other) -> "Vector3":
        self._v += self._coerce(other)
        return self

    def __sub__(self, other) -> "Vector3":
        return Vector3(self._v - self._coerce(other))

    def __rsub__(self, other) -> "Vector3":
        return Vector3(self._coerce(other) - self._v)

    def __isub__(self, other) -> "Vector3":
        self._v -= self._coerce(other)
        return self

    def __mul__(self, scalar) -> "Vector3":
        return Vector3(self._v * float(scalar))

    def __rmul__(self, scalar) -> "Vector3":
        return Vector3(float(scalar) * self._v)

    def __imul__(self, scalar) -> "Vector3":
        self._v *= float(scalar)
        return self

    def __truediv__(self, scalar) -> "Vector3":
        return Vector3(self._v / float(scalar))

    def __itruediv__(self, scalar) -> "Vector3":
        self._v /= float(scalar)
        return self

    def __matmul__(self, other) -> float:
        """Dot product: ``v @ w``."""
        return self.dot(other)

    # ── Vector operations ─────────────────────────────────────────────────────

    def dot(self, other: "Vector3") -> float:
        """Scalar dot product."""
        return float(np.dot(self._v, self._coerce(other)))

    def cross(self, other: "Vector3") -> "Vector3":
        """Cross product: self × other."""
        return Vector3(np.cross(self._v, self._coerce(other)))

    def unit(self) -> "Vector3":
        """Unit vector in the same direction.

        Raises ValueError for near-zero vectors.
        """
        n = self.norm
        if n < 1e-15:
            raise ValueError("Cannot normalize a near-zero vector.")
        return Vector3(self._v / n)

    def angle(self, other: "Vector3") -> float:
        """Angle between this vector and *other* [rad]."""
        cos_theta = self.dot(other) / (self.norm * Vector3._norm(other))
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def project(self, onto: "Vector3") -> "Vector3":
        """Vector projection of self onto *onto*."""
        b = self._coerce(onto)
        return Vector3(np.dot(self._v, b) / np.dot(b, b) * b)

    @staticmethod
    def _norm(v) -> float:
        if isinstance(v, Vector3):
            return v.norm
        return float(np.linalg.norm(v))

    # ── Rotations (active, right-hand rule) ───────────────────────────────────

    def rotx(self, angle: float) -> "Vector3":
        """Rotate by *angle* [rad] about the X axis.

        Active rotation (rotates the vector, not the frame).
        """
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0,   c,  -s],
                      [0.0,   s,   c]])
        return Vector3(R @ self._v)

    def roty(self, angle: float) -> "Vector3":
        """Rotate by *angle* [rad] about the Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[  c, 0.0,   s],
                      [0.0, 1.0, 0.0],
                      [ -s, 0.0,   c]])
        return Vector3(R @ self._v)

    def rotz(self, angle: float) -> "Vector3":
        """Rotate by *angle* [rad] about the Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[  c,  -s, 0.0],
                      [  s,   c, 0.0],
                      [0.0, 0.0, 1.0]])
        return Vector3(R @ self._v)

    def rot(self, axis: "Vector3", angle: float) -> "Vector3":
        """Rotate by *angle* [rad] about an arbitrary *axis* (Rodrigues)."""
        k = self._coerce(axis)
        k = k / np.linalg.norm(k)
        v = self._v
        return Vector3(
            v * np.cos(angle)
            + np.cross(k, v) * np.sin(angle)
            + k * np.dot(k, v) * (1.0 - np.cos(angle))
        )

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_array(self) -> NDArray:
        """Return a copy as a (3,) numpy float64 array."""
        return self._v.copy()

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_array(cls, arr) -> "Vector3":
        """Construct from any array-like of length 3."""
        return cls(arr)
