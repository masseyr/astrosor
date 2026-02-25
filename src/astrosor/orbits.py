"""
orbital_frames.orbits — Keplerian Orbital Mechanics
=====================================================

Two-body orbital mechanics: Keplerian element ↔ ECI state vector conversion,
Kepler equation solver, and basic orbit propagation.  All pure NumPy.
"""

import numpy as np
from numpy.typing import NDArray

from .utils import MU_EARTH, R_EARTH, J2, normalize


# ════════════════════════════════════════════════════════════════════════════
#  Kepler Equation
# ════════════════════════════════════════════════════════════════════════════

def _solve_kepler(M: float, e: float, tol: float = 1e-12,
                  max_iter: int = 50) -> float:
    """Solve Kepler's equation  M = E − e sin(E)  via Newton–Raphson.

    Parameters
    ----------
    M : float — mean anomaly [rad]
    e : float — eccentricity
    tol : float — convergence tolerance [rad]

    Returns
    -------
    E : float — eccentric anomaly [rad]
    """
    # Smart initial guess (Markley-style)
    E = M + 0.85 * e * np.sign(np.sin(M)) if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def _solve_kepler_hyp(M: float, e: float, tol: float = 1e-12,
                      max_iter: int = 50) -> float:
    """Solve the hyperbolic Kepler equation  M = e sinh(H) − H."""
    H = M  # initial guess
    for _ in range(max_iter):
        f = e * np.sinh(H) - H - M
        fp = e * np.cosh(H) - 1.0
        dH = -f / fp
        H += dH
        if abs(dH) < tol:
            break
    return H


# ════════════════════════════════════════════════════════════════════════════
#  Keplerian ↔ ECI Conversions
# ════════════════════════════════════════════════════════════════════════════

def keplerian_to_eci(
    a: float, e: float, i: float,
    raan: float, argp: float, nu: float,
    mu: float = MU_EARTH,
) -> tuple[NDArray, NDArray]:
    """Convert classical Keplerian elements to ECI state vector.

    Parameters
    ----------
    a : float — semi-major axis [m]
    e : float — eccentricity
    i : float — inclination [rad]
    raan : float — right ascension of ascending node [rad]
    argp : float — argument of periapsis [rad]
    nu : float — true anomaly [rad]
    mu : float — gravitational parameter [m³/s²]

    Returns
    -------
    r_eci : (3,) ndarray — position [m]
    v_eci : (3,) ndarray — velocity [m/s]
    """
    p = a * (1.0 - e**2)               # semi-latus rectum
    r_mag = p / (1.0 + e * np.cos(nu))

    # Position & velocity in perifocal (PQW) frame
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix PQW → ECI
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)
    cos_i, sin_i = np.cos(i), np.sin(i)

    R = np.array([
        [cos_raan * cos_argp - sin_raan * sin_argp * cos_i,
         -cos_raan * sin_argp - sin_raan * cos_argp * cos_i,
         sin_raan * sin_i],
        [sin_raan * cos_argp + cos_raan * sin_argp * cos_i,
         -sin_raan * sin_argp + cos_raan * cos_argp * cos_i,
         -cos_raan * sin_i],
        [sin_argp * sin_i,
         cos_argp * sin_i,
         cos_i],
    ])

    return R @ r_pqw, R @ v_pqw


def eci_to_keplerian(
    r_eci: NDArray, v_eci: NDArray, mu: float = MU_EARTH
) -> dict:
    """Convert ECI state vector to classical Keplerian elements.

    Returns
    -------
    dict with keys: a, e, i, raan, argp, nu, E (eccentric anomaly),
    M (mean anomaly), p (semi-latus rectum), h_mag (specific angular momentum)
    All angles in [rad].
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Node vector
    K = np.array([0.0, 0.0, 1.0])
    n = np.cross(K, h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # Semi-major axis (energy)
    xi = v_mag**2 / 2.0 - mu / r_mag
    if abs(1.0 - e) > 1e-10:
        a = -mu / (2.0 * xi)
    else:
        a = np.inf  # parabolic

    # Semi-latus rectum
    p = h_mag**2 / mu

    # Inclination
    inc = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    # RAAN
    if n_mag > 1e-12:
        raan = np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0))
        if n[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    # Argument of periapsis
    if n_mag > 1e-12 and e > 1e-12:
        argp = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    # True anomaly
    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        # Circular orbit — use argument of latitude
        if n_mag > 1e-12:
            nu = np.arccos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1.0, 1.0))
            if r[2] < 0.0:
                nu = 2.0 * np.pi - nu
        else:
            nu = np.arccos(np.clip(r[0] / r_mag, -1.0, 1.0))
            if r[1] < 0.0:
                nu = 2.0 * np.pi - nu

    # Eccentric and mean anomalies
    if e < 1.0:
        E_anom = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(nu / 2.0),
            np.sqrt(1.0 + e) * np.cos(nu / 2.0),
        )
        M_anom = E_anom - e * np.sin(E_anom)
    else:
        E_anom = 0.0
        M_anom = 0.0

    return {
        "a": a, "e": e, "i": inc, "raan": raan, "argp": argp, "nu": nu,
        "E": E_anom, "M": M_anom, "p": p, "h_mag": h_mag,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Orbit Propagation
# ════════════════════════════════════════════════════════════════════════════

def compute_orbital_period(a: float, mu: float = MU_EARTH) -> float:
    """Orbital period [s] for semi-major axis a [m]."""
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def compute_mean_motion(a: float, mu: float = MU_EARTH) -> float:
    """Mean motion [rad/s] for semi-major axis a [m]."""
    return np.sqrt(mu / a**3)


def propagate_kepler(
    r0_eci: NDArray, v0_eci: NDArray, dt: float,
    mu: float = MU_EARTH
) -> tuple[NDArray, NDArray]:
    """Propagate an orbit by dt seconds using the two-body Kepler problem.

    Parameters
    ----------
    r0_eci, v0_eci : (3,) — initial state in ECI
    dt : float — propagation time [s] (can be negative)
    mu : float — gravitational parameter

    Returns
    -------
    r1_eci, v1_eci : (3,) — propagated state
    """
    oe = eci_to_keplerian(r0_eci, v0_eci, mu)
    a, e = oe["a"], oe["e"]

    if e >= 1.0:
        raise NotImplementedError("Hyperbolic/parabolic propagation not yet implemented.")

    n = compute_mean_motion(a, mu)
    M0 = oe["M"]
    M1 = M0 + n * dt
    M1 = M1 % (2.0 * np.pi)

    E1 = _solve_kepler(M1, e)

    # True anomaly from eccentric anomaly
    nu1 = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E1 / 2.0),
        np.sqrt(1.0 - e) * np.cos(E1 / 2.0),
    )

    return keplerian_to_eci(a, e, oe["i"], oe["raan"], oe["argp"], nu1, mu)


def propagate_j2(
    r0_eci: NDArray, v0_eci: NDArray, dt: float,
    mu: float = MU_EARTH, n_steps: int = 100,
) -> tuple[NDArray, NDArray]:
    """Propagate orbit with J2 secular perturbations on RAAN and argp.

    Uses Keplerian propagation with secular drift rates applied to the
    angular elements.  Good for coverage analysis over many orbits.

    Parameters
    ----------
    r0_eci, v0_eci : (3,) — initial state
    dt : float — propagation time [s]
    n_steps : int — unused (analytic, included for API compat)

    Returns
    -------
    r1_eci, v1_eci : (3,) — propagated state
    """
    oe = eci_to_keplerian(r0_eci, v0_eci, mu)
    a, e, inc = oe["a"], oe["e"], oe["i"]
    n = compute_mean_motion(a, mu)
    p = a * (1.0 - e**2)

    # J2 secular rates (Brouwer)
    cos_i = np.cos(inc)
    factor = -1.5 * n * J2 * (R_EARTH / p) ** 2

    raan_dot = factor * cos_i
    argp_dot = factor * (2.0 - 2.5 * np.sin(inc) ** 2)

    raan_new = oe["raan"] + raan_dot * dt
    argp_new = oe["argp"] + argp_dot * dt

    # Mean anomaly advance
    M1 = (oe["M"] + n * dt) % (2.0 * np.pi)
    E1 = _solve_kepler(M1, e)
    nu1 = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E1 / 2.0),
        np.sqrt(1.0 - e) * np.cos(E1 / 2.0),
    )

    return keplerian_to_eci(a, e, inc, raan_new, argp_new, nu1, mu)
