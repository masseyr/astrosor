"""
orbital_frames.coverage — Space-Based Sensor Coverage Analysis
===============================================================

Tools for projecting sensor fields of view onto the Earth's surface,
computing ground traces, evaluating access geometry between a sensor
satellite and ground/space targets, and identifying coverage windows.

All sensor boresight/cone definitions are expressed in UNW or TNW frames,
making it natural to define "nadir-pointing" (along −U in UNW) or
"velocity-pointing" (along +T in TNW) sensor orientations.
"""

import numpy as np
from numpy.typing import NDArray

from .utils import (
    R_EARTH, MU_EARTH, normalize,
    eci_to_ecef, ecef_to_lla, julian_date, gmst,
)
from .frames import (
    eci_to_unw_matrix, eci_to_tnw_matrix,
    unw_to_eci_matrix, tnw_to_eci_matrix,
)
from .orbits import propagate_kepler, propagate_j2, compute_orbital_period


# ════════════════════════════════════════════════════════════════════════════
#  Earth-Ray Intersection
# ════════════════════════════════════════════════════════════════════════════

def earth_intersection(
    origin: NDArray, direction: NDArray, r_earth: float = R_EARTH
) -> NDArray | None:
    """Ray-sphere intersection with Earth (spherical model).

    Parameters
    ----------
    origin : (3,) — ray origin in ECI [m]
    direction : (3,) — ray direction in ECI (need not be unit)
    r_earth : float — Earth radius [m]

    Returns
    -------
    point : (3,) ndarray or None — nearest intersection in ECI [m]
    """
    o = np.asarray(origin, dtype=np.float64)
    d = normalize(np.asarray(direction, dtype=np.float64))

    a = np.dot(d, d)  # = 1.0 since d is unit
    b = 2.0 * np.dot(o, d)
    c = np.dot(o, o) - r_earth**2
    disc = b**2 - 4.0 * a * c

    if disc < 0.0:
        return None  # no intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # We want the nearest positive intersection
    if t1 > 0:
        return o + t1 * d
    elif t2 > 0:
        return o + t2 * d
    else:
        return None  # intersection behind the sensor


# ════════════════════════════════════════════════════════════════════════════
#  Sensor Footprint Generation
# ════════════════════════════════════════════════════════════════════════════

def sensor_footprint_unw(
    r_eci: NDArray, v_eci: NDArray,
    boresight_unw: NDArray,
    half_cone_angle: float,
    n_points: int = 36,
    jd: float | None = None,
) -> dict:
    """Compute the ground footprint of a conical sensor defined in UNW.

    Parameters
    ----------
    r_eci, v_eci : (3,) — satellite state in ECI [m, m/s]
    boresight_unw : (3,) — sensor boresight direction in UNW
        e.g., [-1, 0, 0] for nadir-pointing (−U direction)
    half_cone_angle : float — sensor half-cone angle [rad]
    n_points : int — number of points around the footprint perimeter
    jd : float or None — Julian Date for ECI→ECEF (if None, returns ECI points)

    Returns
    -------
    dict with:
        'eci' : (n_points, 3) — footprint points in ECI (always present)
        'lla' : (n_points, 3) — [lat_rad, lon_rad, alt_m] (only if jd given)
        'center_eci' : (3,) — boresight ground intercept in ECI
        'center_lla' : (3,) — boresight ground intercept LLA (only if jd given)
        'valid' : bool array — True where ray intersected Earth
    """
    R_unw2eci = unw_to_eci_matrix(r_eci, v_eci)
    boresight_eci = R_unw2eci @ normalize(np.asarray(boresight_unw, dtype=np.float64))

    return _compute_footprint(r_eci, boresight_eci, half_cone_angle, n_points, jd)


def sensor_footprint_tnw(
    r_eci: NDArray, v_eci: NDArray,
    boresight_tnw: NDArray,
    half_cone_angle: float,
    n_points: int = 36,
    jd: float | None = None,
) -> dict:
    """Compute the ground footprint of a conical sensor defined in TNW.

    Parameters
    ----------
    boresight_tnw : (3,) — sensor boresight in TNW frame
        e.g., [1, 0, 0] for velocity-direction (along-track) pointing
    (all other params same as sensor_footprint_unw)
    """
    R_tnw2eci = tnw_to_eci_matrix(r_eci, v_eci)
    boresight_eci = R_tnw2eci @ normalize(np.asarray(boresight_tnw, dtype=np.float64))

    return _compute_footprint(r_eci, boresight_eci, half_cone_angle, n_points, jd)


def _compute_footprint(
    r_eci: NDArray, boresight_eci: NDArray,
    half_cone: float, n_points: int, jd: float | None,
) -> dict:
    """Internal: compute footprint from ECI boresight direction."""
    r = np.asarray(r_eci, dtype=np.float64)
    b = normalize(boresight_eci)

    # Build a local frame around the boresight
    # Pick a perpendicular vector
    if abs(np.dot(b, np.array([0, 0, 1]))) < 0.99:
        perp = normalize(np.cross(b, np.array([0.0, 0.0, 1.0])))
    else:
        perp = normalize(np.cross(b, np.array([1.0, 0.0, 0.0])))
    perp2 = np.cross(b, perp)

    # Generate cone edge rays
    azimuths = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    footprint_eci = np.full((n_points, 3), np.nan)
    valid = np.zeros(n_points, dtype=bool)

    for k, az in enumerate(azimuths):
        # Rotate boresight by half_cone about the cone perimeter
        ray_dir = (np.cos(half_cone) * b
                   + np.sin(half_cone) * (np.cos(az) * perp + np.sin(az) * perp2))
        hit = earth_intersection(r, ray_dir)
        if hit is not None:
            footprint_eci[k] = hit
            valid[k] = True

    # Boresight intercept (center of footprint)
    center = earth_intersection(r, b)

    result = {
        "eci": footprint_eci,
        "valid": valid,
        "center_eci": center,
    }

    if jd is not None and center is not None:
        result["center_lla"] = ecef_to_lla(eci_to_ecef(center, jd))
        lla_points = np.full((n_points, 3), np.nan)
        for k in range(n_points):
            if valid[k]:
                lla_points[k] = ecef_to_lla(eci_to_ecef(footprint_eci[k], jd))
        result["lla"] = lla_points

    return result


# ════════════════════════════════════════════════════════════════════════════
#  Ground Trace
# ════════════════════════════════════════════════════════════════════════════

def ground_trace(
    r0_eci: NDArray, v0_eci: NDArray,
    duration: float,
    n_points: int = 360,
    jd_epoch: float = 2451545.0,
    use_j2: bool = True,
    mu: float = MU_EARTH,
) -> NDArray:
    """Compute the sub-satellite ground trace over a time span.

    Parameters
    ----------
    r0_eci, v0_eci : (3,) — initial ECI state
    duration : float — total propagation time [s]
    n_points : int — number of trace points
    jd_epoch : float — Julian Date at t=0
    use_j2 : bool — include J2 secular perturbations
    mu : float — gravitational parameter

    Returns
    -------
    trace : (n_points, 3) ndarray — [lat_rad, lon_rad, alt_m]
    """
    times = np.linspace(0, duration, n_points)
    trace = np.zeros((n_points, 3))
    prop_fn = propagate_j2 if use_j2 else propagate_kepler

    for k, t in enumerate(times):
        r, v = prop_fn(r0_eci, v0_eci, t, mu=mu)
        jd = jd_epoch + t / 86400.0
        r_ecef = eci_to_ecef(r, jd)
        trace[k] = ecef_to_lla(r_ecef)

    return trace


# ════════════════════════════════════════════════════════════════════════════
#  Access Geometry
# ════════════════════════════════════════════════════════════════════════════

def access_geometry(
    r_sensor_eci: NDArray, v_sensor_eci: NDArray,
    r_target_eci: NDArray,
) -> dict:
    """Compute sensor-to-target geometry in both UNW and TNW frames.

    Parameters
    ----------
    r_sensor_eci, v_sensor_eci : (3,) — sensor satellite ECI state
    r_target_eci : (3,) — target position in ECI [m]
        (can be another satellite or a ground point projected to ECI)

    Returns
    -------
    dict with:
        'range' : float — slant range [m]
        'los_eci' : (3,) — line-of-sight unit vector in ECI
        'los_unw' : (3,) — LOS in UNW
        'los_tnw' : (3,) — LOS in TNW
        'elevation' : float — elevation angle from local horizon [rad]
        'azimuth_unw' : float — azimuth in UNW N-W plane [rad]
        'off_boresight_nadir' : float — angle from nadir (−U) [rad]
    """
    rs = np.asarray(r_sensor_eci, dtype=np.float64)
    vs = np.asarray(v_sensor_eci, dtype=np.float64)
    rt = np.asarray(r_target_eci, dtype=np.float64)

    delta = rt - rs
    rng = np.linalg.norm(delta)
    los_eci = delta / rng

    # Transform LOS to local frames
    R_unw = eci_to_unw_matrix(rs, vs)
    R_tnw = eci_to_tnw_matrix(rs, vs)
    los_unw = R_unw @ los_eci
    los_tnw = R_tnw @ los_eci

    # Nadir direction in UNW is −U = [-1, 0, 0]
    nadir_unw = np.array([-1.0, 0.0, 0.0])
    off_nadir = np.arccos(np.clip(np.dot(los_unw, nadir_unw), -1.0, 1.0))

    # Elevation above local horizon
    # Local horizon is perpendicular to U; elevation = 90° − off_nadir
    u_hat = normalize(rs)
    el = np.arcsin(np.clip(-np.dot(los_eci, u_hat), -1.0, 1.0))

    # Azimuth in the N-W plane (0 = along-track N, π/2 = cross-track W)
    az = np.arctan2(los_unw[2], los_unw[1])

    return {
        "range": rng,
        "los_eci": los_eci,
        "los_unw": los_unw,
        "los_tnw": los_tnw,
        "elevation": el,
        "azimuth_unw": az,
        "off_boresight_nadir": off_nadir,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Coverage Windows
# ════════════════════════════════════════════════════════════════════════════

def coverage_windows(
    r0_sensor: NDArray, v0_sensor: NDArray,
    target_eci_fn,
    duration: float,
    half_cone_angle: float,
    boresight_unw: NDArray = np.array([-1.0, 0.0, 0.0]),
    dt_step: float = 10.0,
    jd_epoch: float = 2451545.0,
    use_j2: bool = True,
    mu: float = MU_EARTH,
) -> list[dict]:
    """Find time windows where a target is within the sensor field of view.

    Parameters
    ----------
    r0_sensor, v0_sensor : (3,) — initial sensor satellite ECI state
    target_eci_fn : callable(t) → (3,) ndarray
        Function returning the target ECI position at time t [s].
        For a fixed ground point, wrap ``lla_to_ecef`` + ``ecef_to_eci``.
    duration : float — analysis duration [s]
    half_cone_angle : float — sensor half-cone angle [rad]
    boresight_unw : (3,) — sensor boresight in UNW (default: nadir)
    dt_step : float — time step [s]
    jd_epoch : float — Julian Date at t=0
    use_j2 : bool — J2 perturbations for sensor orbit
    mu : float — gravitational parameter

    Returns
    -------
    windows : list of dict, each with:
        'start' : float — window start time [s from epoch]
        'end' : float — window end time [s from epoch]
        'duration' : float — window duration [s]
        'min_range' : float — closest approach range [m]
        'max_elevation' : float — peak elevation [rad]
    """
    prop_fn = propagate_j2 if use_j2 else propagate_kepler
    boresight = normalize(np.asarray(boresight_unw, dtype=np.float64))

    times = np.arange(0, duration, dt_step)
    in_view = np.zeros(len(times), dtype=bool)
    ranges = np.full(len(times), np.inf)
    elevations = np.full(len(times), -np.pi / 2)

    for k, t in enumerate(times):
        rs, vs = prop_fn(r0_sensor, v0_sensor, t, mu=mu)
        rt = target_eci_fn(t)

        geom = access_geometry(rs, vs, rt)

        # Check if target is within the sensor cone
        R_unw = eci_to_unw_matrix(rs, vs)
        boresight_eci = (unw_to_eci_matrix(rs, vs)) @ boresight
        angle_off = np.arccos(np.clip(
            np.dot(geom["los_eci"], boresight_eci), -1.0, 1.0
        ))

        if angle_off <= half_cone_angle and geom["elevation"] > 0:
            in_view[k] = True
            ranges[k] = geom["range"]
            elevations[k] = geom["elevation"]

    # Extract contiguous windows
    windows = []
    diff = np.diff(in_view.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if in_view[0]:
        starts = np.concatenate([[0], starts])
    if in_view[-1]:
        ends = np.concatenate([ends, [len(times) - 1]])

    for s, e in zip(starts, ends):
        seg = slice(s, e + 1)
        windows.append({
            "start": times[s],
            "end": times[e],
            "duration": times[e] - times[s],
            "min_range": float(np.nanmin(ranges[seg])),
            "max_elevation": float(np.nanmax(elevations[seg])),
        })

    return windows
