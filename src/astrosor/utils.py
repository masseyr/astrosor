"""
orbital_frames.utils — Foundational Utilities
===============================================

Vector math, coordinate conversions (ECI ↔ ECEF ↔ LLA), and time utilities.
All functions are pure NumPy.
"""

import numpy as np
from numpy.typing import NDArray

# ── Physical Constants ──────────────────────────────────────────────────────
MU_EARTH = 3.986004418e14       # Earth gravitational parameter  [m³/s²]
SQRT_MU = 631.3481145928923     
R_EARTH = 6_378_137.0           # WGS-84 semi-major axis          [m]
F_EARTH = 1.0 / 298.257223563  # WGS-84 flattening
E2_EARTH = 2 * F_EARTH - F_EARTH ** 2  # First eccentricity squared
OMEGA_EARTH = 7.2921150e-5      # Earth rotation rate              [rad/s]
J2 = 1.08263e-3                 # J2 zonal harmonic

DAILY_SECONDS = 86400.0
DEFAULT_SIZE = 6.
DEFAULT_RCS = 9.
DEFAULT_RCS_SOURCE = 'UHF'

# ── Vector Helpers ──────────────────────────────────────────────────────────

def normalize(v: NDArray) -> NDArray:
    """Return unit vector.  Works on single vectors or (N,3) arrays."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        mag = np.linalg.norm(v)
        if mag < 1e-15:
            raise ValueError("Cannot normalize a near-zero vector.")
        return v / mag
    elif v.ndim == 2:
        mag = np.linalg.norm(v, axis=1, keepdims=True)
        if np.any(mag < 1e-15):
            raise ValueError("Cannot normalize a near-zero vector.")
        return v / mag
    else:
        raise ValueError(f"Expected 1-D or 2-D array, got {v.ndim}-D.")


def rotation_matrix_axis_angle(axis: NDArray, angle: float) -> NDArray:
    """Rotation matrix via Rodrigues' formula (right-hand, active rotation).

    Parameters
    ----------
    axis : (3,) array — rotation axis (will be normalized internally)
    angle : float — rotation angle [rad]

    Returns
    -------
    R : (3,3) ndarray — rotation matrix
    """
    k = normalize(np.asarray(axis, dtype=np.float64))
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) * c + (1.0 - c) * np.outer(k, k) + s * K


# ── Time Utilities ──────────────────────────────────────────────────────────

def julian_date(year: int, month: int, day: int,
                hour: float = 0.0, minute: float = 0.0,
                second: float = 0.0) -> float:
    """Compute Julian Date from calendar date (UTC)."""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    JD = (int(365.25 * (year + 4716))
          + int(30.6001 * (month + 1))
          + day + B - 1524.5)
    JD += (hour + minute / 60.0 + second / 3600.0) / 24.0
    return JD


def gmst(jd: float) -> float:
    """Greenwich Mean Sidereal Time [rad] from Julian Date.

    Uses the IAU 1982 model (accurate to ~0.1 arcsec for dates near J2000).
    """
    T = (jd - 2_451_545.0) / 36_525.0
    # GMST in seconds of time at 0h UT
    theta_sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T \
                + 0.093104 * T**2 - 6.2e-6 * T**3
    theta_deg = (theta_sec / 240.0) % 360.0  # convert seconds→degrees
    return np.deg2rad(theta_deg)


# ── Coordinate Conversions ──────────────────────────────────────────────────

def eci_to_ecef(r_eci: NDArray, jd: float) -> NDArray:
    """Rotate ECI position vector to ECEF given Julian Date.

    Parameters
    ----------
    r_eci : (3,) or (N,3) array — position in ECI [m]
    jd : float — Julian Date (UTC, ignoring polar motion / UT1−UTC)

    Returns
    -------
    r_ecef : same shape as r_eci [m]
    """
    theta = gmst(jd)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    r_eci = np.asarray(r_eci, dtype=np.float64)
    if r_eci.ndim == 1:
        return R @ r_eci
    return (R @ r_eci.T).T


def ecef_to_eci(r_ecef: NDArray, jd: float) -> NDArray:
    """Rotate ECEF position vector to ECI given Julian Date."""
    theta = gmst(jd)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    r_ecef = np.asarray(r_ecef, dtype=np.float64)
    if r_ecef.ndim == 1:
        return R @ r_ecef
    return (R @ r_ecef.T).T


def ecef_to_lla(r_ecef: NDArray) -> NDArray:
    """ECEF [m] → geodetic Latitude [rad], Longitude [rad], Altitude [m].

    Uses Bowring's iterative method (converges in 2-3 iterations).

    Returns
    -------
    lla : (3,) or (N,3) array — [lat, lon, alt]
    """
    r = np.asarray(r_ecef, dtype=np.float64)
    single = r.ndim == 1
    if single:
        r = r.reshape(1, 3)

    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    # Initial guess using spherical approximation
    lat = np.arctan2(z, p * (1.0 - E2_EARTH))

    for _ in range(5):
        sin_lat = np.sin(lat)
        N_phi = R_EARTH / np.sqrt(1.0 - E2_EARTH * sin_lat**2)
        lat = np.arctan2(z + E2_EARTH * N_phi * sin_lat, p)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N_phi = R_EARTH / np.sqrt(1.0 - E2_EARTH * sin_lat**2)
    alt = p / cos_lat - N_phi
    # Handle polar singularity
    polar = np.abs(cos_lat) < 1e-10
    if np.any(polar):
        alt[polar] = np.abs(z[polar]) - R_EARTH * np.sqrt(1.0 - E2_EARTH)

    result = np.stack([lat, lon, alt], axis=-1)
    return result[0] if single else result


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> NDArray:
    """Geodetic LLA → ECEF [m].

    Parameters
    ----------
    lat, lon : float — geodetic latitude / longitude [rad]
    alt : float — altitude above WGS-84 ellipsoid [m]
    """
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    N = R_EARTH / np.sqrt(1.0 - E2_EARTH * sin_lat**2)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - E2_EARTH) + alt) * sin_lat
    return np.array([x, y, z])
