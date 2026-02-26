"""
orbital_frames.sun — Solar Ephemeris & Solar Exclusion Analysis
================================================================

Low-precision analytical solar position (accurate to ~1° over ±50 years
from J2000) and tools for space-based sensor solar exclusion analysis.

Capabilities
------------
- Sun position in ECI (geocentric equatorial)
- Solar exclusion zone checks (keep-out cone around Sun)
- Earth shadow geometry (cylindrical and conical umbra/penumbra models)
- Eclipse state classification (sunlit / penumbra / umbra)
- Solar phase angle for target illumination analysis
- Sun-Earth-Probe (SEP) angle

All functions are pure NumPy.  The solar ephemeris follows the
low-precision formulae from Meeus (1998) / Astronomical Almanac,
sufficient for sensor exclusion and illumination analysis.

Reference
---------
Meeus, J. (1998). *Astronomical Algorithms*, 2nd ed., Willmann-Bell.
Vallado, D.A. (2013). *Fundamentals of Astrodynamics*, 4th ed., §5.1.
"""

import numpy as np
from numpy.typing import NDArray

from .utils import normalize, R_EARTH

# ── Constants ───────────────────────────────────────────────────────────────
AU = 149_597_870_700.0          # Astronomical Unit [m]
R_SUN = 696_000_000.0           # Solar radius [m]
SOLAR_FLUX_1AU = 1361.0         # Solar irradiance at 1 AU [W/m²]


# ════════════════════════════════════════════════════════════════════════════
#  Solar Ephemeris
# ════════════════════════════════════════════════════════════════════════════

def sun_position_eci(jd: float) -> NDArray:
    """Compute geocentric Sun position in ECI (J2000 equatorial).

    Uses the low-precision solar ephemeris from the Astronomical Almanac
    (accurate to ~0.01° in ecliptic longitude, ~1' in obliquity).

    Parameters
    ----------
    jd : float — Julian Date (TDB ≈ UTC for this precision)

    Returns
    -------
    r_sun : (3,) ndarray — Sun position vector in ECI [m]
    """
    # Julian centuries from J2000.0
    T = (jd - 2_451_545.0) / 36_525.0

    # Mean anomaly of the Sun [deg]
    M = 357.5291092 + 35999.0502909 * T
    M = np.deg2rad(M % 360.0)

    # Mean longitude of the Sun [deg]
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    L0 = L0 % 360.0

    # Equation of center [deg]
    C = (1.9146 - 0.004817 * T - 0.000014 * T**2) * np.sin(M) \
      + (0.019993 - 0.000101 * T) * np.sin(2 * M) \
      + 0.00029 * np.sin(3 * M)

    # Sun's true longitude [deg]
    sun_lon = np.deg2rad((L0 + C) % 360.0)

    # Sun's distance [AU]
    e_sun = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    nu = M + np.deg2rad(C)
    R_au = 1.000001018 * (1.0 - e_sun**2) / (1.0 + e_sun * np.cos(nu))

    # Mean obliquity of ecliptic [deg]
    eps = 23.439291 - 0.0130042 * T - 1.64e-7 * T**2 + 5.04e-7 * T**3
    eps = np.deg2rad(eps)

    # Ecliptic → equatorial (ECI)
    r_m = R_au * AU
    x = r_m * np.cos(sun_lon)
    y = r_m * np.cos(eps) * np.sin(sun_lon)
    z = r_m * np.sin(eps) * np.sin(sun_lon)

    return np.array([x, y, z])


def sun_direction_eci(jd: float) -> NDArray:
    """Unit vector from Earth to Sun in ECI."""
    return normalize(sun_position_eci(jd))


def sun_distance(jd: float) -> float:
    """Earth-Sun distance [m]."""
    return float(np.linalg.norm(sun_position_eci(jd)))


def solar_declination_ra(jd: float) -> tuple[float, float]:
    """Solar right ascension and declination [rad].

    Returns
    -------
    ra : float — right ascension [rad, 0..2π]
    dec : float — declination [rad, -π/2..π/2]
    """
    r = sun_position_eci(jd)
    ra = np.arctan2(r[1], r[0]) % (2 * np.pi)
    dec = np.arcsin(np.clip(r[2] / np.linalg.norm(r), -1.0, 1.0))
    return ra, dec


# ════════════════════════════════════════════════════════════════════════════
#  Solar Exclusion Zone
# ════════════════════════════════════════════════════════════════════════════

def sun_angle(r_sat_eci: NDArray, boresight_eci: NDArray,
              jd: float) -> float:
    """Angle between sensor boresight and the Sun direction [rad].

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    boresight_eci : (3,) — sensor boresight unit vector in ECI
    jd : float — Julian Date

    Returns
    -------
    angle : float — angular separation [rad]
    """
    sun_dir = sun_direction_eci(jd)
    b = normalize(np.asarray(boresight_eci, dtype=np.float64))
    return np.arccos(np.clip(np.dot(b, sun_dir), -1.0, 1.0))


def check_solar_exclusion(r_sat_eci: NDArray, boresight_eci: NDArray,
                          jd: float, exclusion_half_angle: float) -> dict:
    """Check if a sensor boresight violates the solar exclusion zone.

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    boresight_eci : (3,) — sensor boresight unit vector in ECI
    jd : float — Julian Date
    exclusion_half_angle : float — solar keep-out half-angle [rad]
        Typical values: 30°–45° for optical sensors, 15°–20° for radar.

    Returns
    -------
    dict with:
        'excluded' : bool — True if boresight is inside the exclusion zone
        'sun_angle' : float — angle between boresight and Sun [rad]
        'margin' : float — angular margin to exclusion boundary [rad]
            Positive = safe, negative = violated.
        'sun_direction_eci' : (3,) — Sun unit vector in ECI
    """
    angle = sun_angle(r_sat_eci, boresight_eci, jd)
    margin = angle - exclusion_half_angle
    return {
        "excluded": bool(angle < exclusion_half_angle),
        "sun_angle": float(angle),
        "margin": float(margin),
        "sun_direction_eci": sun_direction_eci(jd),
    }


def solar_exclusion_windows(
    r0_eci: NDArray, v0_eci: NDArray,
    boresight_body_fn,
    exclusion_half_angle: float,
    duration: float,
    dt_step: float = 30.0,
    jd_epoch: float = 2451545.0,
    propagate_fn=None,
) -> list[dict]:
    """Find time windows where the sensor boresight is in solar exclusion.

    Parameters
    ----------
    r0_eci, v0_eci : (3,) — initial satellite ECI state
    boresight_body_fn : callable(r_eci, v_eci) → (3,) ndarray
        Returns the sensor boresight unit vector in ECI given the current
        satellite state.  For a nadir-pointing sensor::

            lambda r, v: -normalize(r)

    exclusion_half_angle : float — solar keep-out half-angle [rad]
    duration : float — analysis span [s]
    dt_step : float — time step [s]
    jd_epoch : float — Julian Date at t=0
    propagate_fn : callable or None — orbit propagator (default: propagate_kepler)

    Returns
    -------
    windows : list of dict with 'start', 'end', 'duration',
        'min_sun_angle' (closest approach to Sun [rad])
    """
    from .orbits import propagate_kepler
    prop = propagate_fn or propagate_kepler

    times = np.arange(0, duration, dt_step)
    excluded = np.zeros(len(times), dtype=bool)
    angles = np.full(len(times), np.pi)

    for k, t in enumerate(times):
        r, v = prop(r0_eci, v0_eci, t)
        jd = jd_epoch + t / 86400.0
        bore = boresight_body_fn(r, v)
        ang = sun_angle(r, bore, jd)
        angles[k] = ang
        excluded[k] = ang < exclusion_half_angle

    # Extract contiguous windows
    windows = []
    diff = np.diff(excluded.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if excluded[0]:
        starts = np.concatenate([[0], starts])
    if excluded[-1]:
        ends = np.concatenate([ends, [len(times) - 1]])

    for s, e in zip(starts, ends):
        seg = slice(s, e + 1)
        windows.append({
            "start": float(times[s]),
            "end": float(times[e]),
            "duration": float(times[e] - times[s]),
            "min_sun_angle": float(np.min(angles[seg])),
        })
    return windows


# ════════════════════════════════════════════════════════════════════════════
#  Earth Shadow / Eclipse Geometry
# ════════════════════════════════════════════════════════════════════════════

def eclipse_cylindrical(r_sat_eci: NDArray, jd: float) -> str:
    """Determine eclipse state using the cylindrical shadow model.

    This is the simplest model: Earth casts a cylinder of radius R_⊕
    in the anti-Sun direction.  No penumbra distinction.

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    jd : float — Julian Date

    Returns
    -------
    state : str — 'sunlit' or 'eclipse'
    """
    r = np.asarray(r_sat_eci, dtype=np.float64)
    sun_hat = sun_direction_eci(jd)

    # Project satellite onto the Sun direction
    proj = np.dot(r, sun_hat)

    if proj > 0:
        # Satellite is on the sunward side — always sunlit
        return "sunlit"

    # Perpendicular distance from Earth-Sun line
    perp = np.linalg.norm(r - proj * sun_hat)

    if perp < R_EARTH:
        return "eclipse"
    return "sunlit"


def eclipse_conical(r_sat_eci: NDArray, jd: float) -> dict:
    """Determine eclipse state using the conical shadow model.

    Models both the umbra (total shadow) and penumbra (partial shadow)
    cast by Earth as cones, accounting for the finite size of the Sun.

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    jd : float — Julian Date

    Returns
    -------
    dict with:
        'state' : str — 'sunlit', 'penumbra', or 'umbra'
        'shadow_fraction' : float — 0.0 (full sun) to 1.0 (full umbra)
        'penumbra_depth' : float — fractional depth into penumbra [0..1]
    """
    r = np.asarray(r_sat_eci, dtype=np.float64)
    r_sun = sun_position_eci(jd)
    d_sun = np.linalg.norm(r_sun)

    # Unit vector from Earth to Sun
    s_hat = r_sun / d_sun

    # Umbra and penumbra half-angles
    alpha_umbra = np.arcsin((R_SUN - R_EARTH) / d_sun)     # umbra cone half-angle
    alpha_penumbra = np.arcsin((R_SUN + R_EARTH) / d_sun)   # penumbra cone half-angle

    # Satellite distance from Earth-Sun line
    r_mag = np.linalg.norm(r)
    cos_theta = np.dot(r, s_hat) / r_mag
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # angle from Sun direction

    # Only check shadow if satellite is on anti-Sun side
    if cos_theta > 0:
        return {"state": "sunlit", "shadow_fraction": 0.0, "penumbra_depth": 0.0}

    # Perpendicular distance from Earth-Sun axis
    perp_dist = r_mag * np.sin(theta)

    # Project satellite onto anti-Sun axis (distance behind Earth)
    proj_dist = -r_mag * cos_theta  # positive behind Earth

    # Umbra and penumbra radii at the satellite's projected distance
    # Umbra cone narrows; penumbra cone widens
    r_umbra = R_EARTH - proj_dist * np.tan(alpha_umbra)
    r_penumbra = R_EARTH + proj_dist * np.tan(alpha_penumbra)

    if r_umbra > 0 and perp_dist < r_umbra:
        return {"state": "umbra", "shadow_fraction": 1.0, "penumbra_depth": 1.0}
    elif perp_dist < r_penumbra:
        # Linear interpolation through the penumbra
        if r_umbra > 0:
            depth = (r_penumbra - perp_dist) / (r_penumbra - r_umbra)
        else:
            depth = (r_penumbra - perp_dist) / r_penumbra
        depth = np.clip(depth, 0.0, 1.0)
        frac = depth  # simplified linear shadow fraction
        return {"state": "penumbra", "shadow_fraction": float(frac),
                "penumbra_depth": float(depth)}
    else:
        return {"state": "sunlit", "shadow_fraction": 0.0, "penumbra_depth": 0.0}


def eclipse_intervals(
    r0_eci: NDArray, v0_eci: NDArray,
    duration: float,
    dt_step: float = 30.0,
    jd_epoch: float = 2451545.0,
    model: str = "conical",
    propagate_fn=None,
) -> list[dict]:
    """Find eclipse intervals over a time span.

    Parameters
    ----------
    r0_eci, v0_eci : (3,) — initial satellite ECI state
    duration : float — analysis span [s]
    dt_step : float — time step [s]
    jd_epoch : float — Julian Date at t=0
    model : str — 'cylindrical' or 'conical'
    propagate_fn : callable or None — propagator (default: propagate_kepler)

    Returns
    -------
    intervals : list of dict with:
        'start', 'end', 'duration' : float — times [s from epoch]
        'type' : str — 'eclipse' (cylindrical) or 'umbra'/'penumbra' (conical)
        'max_shadow_fraction' : float — peak shadow (conical only)
    """
    from .orbits import propagate_kepler
    prop = propagate_fn or propagate_kepler

    times = np.arange(0, duration, dt_step)
    in_shadow = np.zeros(len(times), dtype=bool)
    shadow_fracs = np.zeros(len(times))

    for k, t in enumerate(times):
        r, v = prop(r0_eci, v0_eci, t)
        jd = jd_epoch + t / 86400.0
        if model == "cylindrical":
            state = eclipse_cylindrical(r, jd)
            in_shadow[k] = state == "eclipse"
            shadow_fracs[k] = 1.0 if in_shadow[k] else 0.0
        else:
            result = eclipse_conical(r, jd)
            in_shadow[k] = result["state"] != "sunlit"
            shadow_fracs[k] = result["shadow_fraction"]

    # Extract contiguous intervals
    intervals = []
    diff = np.diff(in_shadow.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if in_shadow[0]:
        starts = np.concatenate([[0], starts])
    if in_shadow[-1]:
        ends = np.concatenate([ends, [len(times) - 1]])

    for s, e in zip(starts, ends):
        seg = slice(s, e + 1)
        intervals.append({
            "start": float(times[s]),
            "end": float(times[e]),
            "duration": float(times[e] - times[s]),
            "type": "eclipse" if model == "cylindrical" else "umbra/penumbra",
            "max_shadow_fraction": float(np.max(shadow_fracs[seg])),
        })
    return intervals


# ════════════════════════════════════════════════════════════════════════════
#  Illumination & Phase Angle
# ════════════════════════════════════════════════════════════════════════════

def solar_phase_angle(r_sat_eci: NDArray, r_target_eci: NDArray,
                      jd: float) -> float:
    """Solar phase angle at the target as seen from the sensor satellite.

    The phase angle is the angle Sun–Target–Sensor.  At 0° the target is
    fully illuminated (backlit); at 180° the sensor looks directly at the
    dark side.

    Parameters
    ----------
    r_sat_eci : (3,) — sensor satellite position in ECI [m]
    r_target_eci : (3,) — target position in ECI [m]
    jd : float — Julian Date

    Returns
    -------
    phase : float — phase angle [rad, 0..π]
    """
    r_sun = sun_position_eci(jd)
    t = np.asarray(r_target_eci, dtype=np.float64)

    # Vectors from target to Sun and from target to sensor
    to_sun = normalize(r_sun - t)
    to_sensor = normalize(np.asarray(r_sat_eci, dtype=np.float64) - t)

    return np.arccos(np.clip(np.dot(to_sun, to_sensor), -1.0, 1.0))


def is_target_illuminated(r_target_eci: NDArray, jd: float,
                          model: str = "cylindrical") -> bool:
    """Check whether a target position is in sunlight.

    Parameters
    ----------
    r_target_eci : (3,) — target position in ECI [m]
    jd : float — Julian Date
    model : str — 'cylindrical' or 'conical'

    Returns
    -------
    illuminated : bool
    """
    if model == "cylindrical":
        return eclipse_cylindrical(r_target_eci, jd) == "sunlit"
    else:
        return eclipse_conical(r_target_eci, jd)["state"] == "sunlit"


def sep_angle(r_sat_eci: NDArray, jd: float) -> float:
    """Sun-Earth-Probe (SEP) angle [rad].

    The angle at Earth between the Sun and the satellite.
    Important for deep-space comms and solar conjunction geometry.
    """
    sun_hat = sun_direction_eci(jd)
    sat_hat = normalize(np.asarray(r_sat_eci, dtype=np.float64))
    return np.arccos(np.clip(np.dot(sun_hat, sat_hat), -1.0, 1.0))
