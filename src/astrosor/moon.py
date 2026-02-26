"""
orbital_frames.moon — Lunar Ephemeris & Lunar Exclusion Analysis
=================================================================

Low-precision analytical lunar position (accurate to ~0.3° in longitude,
~0.2° in latitude over ±50 years from J2000) and tools for space-based
sensor lunar exclusion analysis.

Capabilities
------------
- Moon position in ECI (geocentric equatorial)
- Lunar exclusion zone checks (keep-out cone around Moon)
- Moon illumination (phase, fraction, age)
- Lunar glint geometry for ground/sea sensor analysis
- Sun–Moon angular separation (conjunction / opposition)

All functions are pure NumPy.  The lunar ephemeris follows the truncated
Brown theory (principal terms) as presented in Meeus (1998).

Reference
---------
Meeus, J. (1998). *Astronomical Algorithms*, 2nd ed., Willmann-Bell, Ch. 47.
Montenbruck, O. & Gill, E. (2000). *Satellite Orbits*, §3.3.2.
"""

import numpy as np
from numpy.typing import NDArray

from .utils import normalize, R_EARTH

# ── Constants ───────────────────────────────────────────────────────────────
R_MOON = 1_737_400.0                   # Mean lunar radius [m]
MEAN_EARTH_MOON_DIST = 384_400_000.0   # Mean Earth-Moon distance [m]


# ════════════════════════════════════════════════════════════════════════════
#  Lunar Ephemeris
# ════════════════════════════════════════════════════════════════════════════

def moon_position_eci(jd: float) -> NDArray:
    """Compute geocentric Moon position in ECI (J2000 equatorial).

    Uses the principal periodic terms of the truncated Brown lunar theory
    as presented in Meeus (1998, Ch. 47) and Montenbruck & Gill (2000).
    Accuracy: ~0.3° longitude, ~0.2° latitude, ~0.5% distance.

    Parameters
    ----------
    jd : float — Julian Date (TDB ≈ UTC for this precision)

    Returns
    -------
    r_moon : (3,) ndarray — Moon position vector in ECI [m]
    """
    T = (jd - 2_451_545.0) / 36_525.0  # Julian centuries from J2000

    # Fundamental arguments [deg]
    # L' — Moon's mean longitude
    Lp = 218.3164477 + 481267.88123421 * T \
         - 0.0015786 * T**2 + T**3 / 538841.0 - T**4 / 65194000.0

    # D — Mean elongation of the Moon
    D = 297.8501921 + 445267.1114034 * T \
        - 0.0018819 * T**2 + T**3 / 545868.0 - T**4 / 113065000.0

    # M — Sun's mean anomaly
    M = 357.5291092 + 35999.0502909 * T \
        - 0.0001536 * T**2 + T**3 / 24490000.0

    # M' — Moon's mean anomaly
    Mp = 134.9633964 + 477198.8675055 * T \
         + 0.0087414 * T**2 + T**3 / 69699.0 - T**4 / 14712000.0

    # F — Moon's argument of latitude
    F = 93.2720950 + 483202.0175233 * T \
        - 0.0036539 * T**2 - T**3 / 3526000.0 + T**4 / 863310000.0

    # Convert to radians
    Lp_r = np.deg2rad(Lp % 360.0)
    D_r = np.deg2rad(D % 360.0)
    M_r = np.deg2rad(M % 360.0)
    Mp_r = np.deg2rad(Mp % 360.0)
    F_r = np.deg2rad(F % 360.0)

    # ── Longitude terms (principal) ──
    # (D, M, M', F, coeff_sin [1e-6 deg])
    lon_terms = [
        (0, 0, 1, 0, 6288774),
        (2, 0, -1, 0, 1274027),
        (2, 0, 0, 0, 658314),
        (0, 0, 2, 0, 213618),
        (0, 1, 0, 0, -185116),
        (0, 0, 0, 2, -114332),
        (2, 0, -2, 0, 58793),
        (2, -1, -1, 0, 57066),
        (2, 0, 1, 0, 53322),
        (2, -1, 0, 0, 45758),
        (0, 1, -1, 0, -40923),
        (1, 0, 0, 0, -34720),
        (0, 1, 1, 0, -30383),
        (2, 0, 0, -2, 15327),
        (0, 0, 1, 2, -12528),
        (0, 0, 1, -2, 10980),
        (4, 0, -1, 0, 10675),
        (0, 0, 3, 0, 10034),
        (4, 0, -2, 0, 8548),
        (2, 1, -1, 0, -7888),
    ]

    # ── Latitude terms (principal) ──
    lat_terms = [
        (0, 0, 0, 1, 5128122),
        (0, 0, 1, 1, 280602),
        (0, 0, 1, -1, 277693),
        (2, 0, 0, -1, 173237),
        (2, 0, -1, 1, 55413),
        (2, 0, -1, -1, 46271),
        (2, 0, 0, 1, 32573),
        (0, 0, 2, 1, 17198),
        (2, 0, 1, -1, 9266),
        (0, 0, 2, -1, 8822),
        (2, -1, 0, -1, 8216),
        (2, 0, -2, -1, 4324),
        (2, 0, 1, 1, 4200),
        (2, 1, 0, -1, -3359),
        (2, -1, -1, 1, 2463),
        (2, -1, 0, 1, 2211),
        (2, -1, -1, -1, 2065),
        (0, 1, -1, -1, -1870),
    ]

    # ── Distance terms (principal) ──
    # (D, M, M', F, coeff_cos [m adjusted from km])
    dist_terms = [
        (0, 0, 1, 0, -20905355),
        (2, 0, -1, 0, -3699111),
        (2, 0, 0, 0, -2955968),
        (0, 0, 2, 0, -569925),
        (0, 1, 0, 0, 48888),
        (0, 0, 0, 2, -3149),
        (2, 0, -2, 0, 246158),
        (2, -1, -1, 0, -152138),
        (2, 0, 1, 0, -170733),
        (2, -1, 0, 0, -204586),
        (0, 1, -1, 0, -129620),
        (1, 0, 0, 0, 108743),
        (0, 1, 1, 0, 104755),
        (2, 0, 0, -2, 10321),
    ]

    # Sum longitude
    sum_l = 0.0
    for d, m, mp, f, coeff in lon_terms:
        arg = d * D_r + m * M_r + mp * Mp_r + f * F_r
        sum_l += coeff * np.sin(arg)
    lam = Lp + sum_l * 1e-6  # ecliptic longitude [deg]

    # Sum latitude
    sum_b = 0.0
    for d, m, mp, f, coeff in lat_terms:
        arg = d * D_r + m * M_r + mp * Mp_r + f * F_r
        sum_b += coeff * np.sin(arg)
    beta = sum_b * 1e-6  # ecliptic latitude [deg]

    # Sum distance
    sum_r = 0.0
    for d, m, mp, f, coeff in dist_terms:
        arg = d * D_r + m * M_r + mp * Mp_r + f * F_r
        sum_r += coeff * np.cos(arg)
    dist = 385000.56 + sum_r * 1e-3  # distance [km]
    dist_m = dist * 1000.0  # convert to meters

    # Ecliptic to equatorial
    lam_r = np.deg2rad(lam % 360.0)
    beta_r = np.deg2rad(beta)

    # Mean obliquity
    eps = 23.439291 - 0.0130042 * T
    eps_r = np.deg2rad(eps)

    cos_b = np.cos(beta_r)
    x_ecl = dist_m * cos_b * np.cos(lam_r)
    y_ecl = dist_m * cos_b * np.sin(lam_r)
    z_ecl = dist_m * np.sin(beta_r)

    # Rotate ecliptic → equatorial (ECI)
    cos_e, sin_e = np.cos(eps_r), np.sin(eps_r)
    x = x_ecl
    y = cos_e * y_ecl - sin_e * z_ecl
    z = sin_e * y_ecl + cos_e * z_ecl

    return np.array([x, y, z])


def moon_direction_eci(jd: float) -> NDArray:
    """Unit vector from Earth to Moon in ECI."""
    return normalize(moon_position_eci(jd))


def moon_distance(jd: float) -> float:
    """Earth-Moon distance [m]."""
    return float(np.linalg.norm(moon_position_eci(jd)))


# ════════════════════════════════════════════════════════════════════════════
#  Lunar Exclusion Zone
# ════════════════════════════════════════════════════════════════════════════

def moon_angle(r_sat_eci: NDArray, boresight_eci: NDArray,
               jd: float) -> float:
    """Angle between sensor boresight and the Moon direction [rad].

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    boresight_eci : (3,) — sensor boresight unit vector in ECI
    jd : float — Julian Date

    Returns
    -------
    angle : float — angular separation [rad]
    """
    moon_dir = moon_direction_eci(jd)
    b = normalize(np.asarray(boresight_eci, dtype=np.float64))
    return np.arccos(np.clip(np.dot(b, moon_dir), -1.0, 1.0))


def moon_angular_radius(jd: float) -> float:
    """Apparent angular radius of the Moon as seen from Earth center [rad]."""
    return np.arcsin(R_MOON / moon_distance(jd))


def check_lunar_exclusion(r_sat_eci: NDArray, boresight_eci: NDArray,
                          jd: float, exclusion_half_angle: float) -> dict:
    """Check if a sensor boresight violates the lunar exclusion zone.

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    boresight_eci : (3,) — sensor boresight unit vector in ECI
    jd : float — Julian Date
    exclusion_half_angle : float — lunar keep-out half-angle [rad]
        Typical: 5°–15° for optical sensors (depends on lunar phase).

    Returns
    -------
    dict with:
        'excluded' : bool — True if boresight is inside exclusion zone
        'moon_angle' : float — angle between boresight and Moon [rad]
        'margin' : float — angular margin [rad] (positive = safe)
        'moon_direction_eci' : (3,) — Moon unit vector in ECI
        'moon_phase_fraction' : float — illuminated fraction [0..1]
    """
    angle = moon_angle(r_sat_eci, boresight_eci, jd)
    margin = angle - exclusion_half_angle
    phase = moon_illumination_fraction(jd)
    return {
        "excluded": bool(angle < exclusion_half_angle),
        "moon_angle": float(angle),
        "margin": float(margin),
        "moon_direction_eci": moon_direction_eci(jd),
        "moon_phase_fraction": float(phase),
    }


def lunar_exclusion_windows(
    r0_eci: NDArray, v0_eci: NDArray,
    boresight_body_fn,
    exclusion_half_angle: float,
    duration: float,
    dt_step: float = 30.0,
    jd_epoch: float = 2451545.0,
    propagate_fn=None,
) -> list[dict]:
    """Find time windows where the sensor boresight is in lunar exclusion.

    Parameters same as sun.solar_exclusion_windows but for the Moon.

    Returns
    -------
    windows : list of dict with 'start', 'end', 'duration',
        'min_moon_angle', 'mean_phase_fraction'
    """
    from .orbits import propagate_kepler
    prop = propagate_fn or propagate_kepler

    times = np.arange(0, duration, dt_step)
    excluded = np.zeros(len(times), dtype=bool)
    angles = np.full(len(times), np.pi)
    phases = np.zeros(len(times))

    for k, t in enumerate(times):
        r, v = prop(r0_eci, v0_eci, t)
        jd = jd_epoch + t / 86400.0
        bore = boresight_body_fn(r, v)
        angles[k] = moon_angle(r, bore, jd)
        excluded[k] = angles[k] < exclusion_half_angle
        phases[k] = moon_illumination_fraction(jd)

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
            "min_moon_angle": float(np.min(angles[seg])),
            "mean_phase_fraction": float(np.mean(phases[seg])),
        })
    return windows


# ════════════════════════════════════════════════════════════════════════════
#  Moon Illumination & Phase
# ════════════════════════════════════════════════════════════════════════════

def moon_phase_angle(jd: float) -> float:
    """Sun-Moon elongation (phase angle) [rad].

    The angle at the Moon between the Sun and the Earth.
    0° = full Moon (opposition), 180° = new Moon (conjunction).
    """
    r_moon = moon_position_eci(jd)
    from .sun import sun_position_eci
    r_sun = sun_position_eci(jd)

    # Vectors from Moon to Sun and Moon to Earth
    moon_to_sun = normalize(r_sun - r_moon)
    moon_to_earth = normalize(-r_moon)

    return np.arccos(np.clip(np.dot(moon_to_sun, moon_to_earth), -1.0, 1.0))


def moon_illumination_fraction(jd: float) -> float:
    """Fraction of the Moon's disk that is illuminated [0..1].

    Uses the simple cos-based formula:  k = (1 + cos(i)) / 2
    where i is the phase angle.
    """
    i = moon_phase_angle(jd)
    return float((1.0 + np.cos(i)) / 2.0)


def moon_age_days(jd: float) -> float:
    """Approximate lunar age (days since last new Moon) [0..29.53].

    Based on mean synodic month.  Accurate to ~±0.5 days.
    """
    SYNODIC_MONTH = 29.530588853  # days
    # Known new Moon reference: 2000 Jan 6 18:14 UTC
    JD_NEW_MOON_REF = 2451550.26
    age = (jd - JD_NEW_MOON_REF) % SYNODIC_MONTH
    return float(age)


def moon_phase_name(jd: float) -> str:
    """Human-readable lunar phase name."""
    age = moon_age_days(jd)
    if age < 1.85:
        return "New Moon"
    elif age < 7.38:
        return "Waxing Crescent"
    elif age < 9.23:
        return "First Quarter"
    elif age < 14.77:
        return "Waxing Gibbous"
    elif age < 16.61:
        return "Full Moon"
    elif age < 22.15:
        return "Waning Gibbous"
    elif age < 23.99:
        return "Last Quarter"
    elif age < 27.68:
        return "Waning Crescent"
    else:
        return "New Moon"


# ════════════════════════════════════════════════════════════════════════════
#  Sun–Moon Geometry
# ════════════════════════════════════════════════════════════════════════════

def sun_moon_angle(jd: float) -> float:
    """Angular separation between Sun and Moon as seen from Earth [rad]."""
    from .sun import sun_direction_eci
    return np.arccos(np.clip(
        np.dot(sun_direction_eci(jd), moon_direction_eci(jd)), -1.0, 1.0
    ))


def moon_earth_shadow(r_target_eci: NDArray, jd: float) -> bool:
    """Check if a target point is in the Moon's shadow (lunar eclipse of sat).

    Uses a cylindrical shadow model for the Moon.

    Parameters
    ----------
    r_target_eci : (3,) — target position in ECI [m]
    jd : float — Julian Date

    Returns
    -------
    in_shadow : bool — True if the target is in the Moon's shadow
    """
    from .sun import sun_position_eci
    r_sun = sun_position_eci(jd)
    r_moon = moon_position_eci(jd)
    r_target = np.asarray(r_target_eci, dtype=np.float64)

    # Sun direction as seen from the Moon
    sun_hat = normalize(r_sun - r_moon)

    # Vector from Moon to target
    delta = r_target - r_moon

    # Project onto anti-Sun direction
    proj = np.dot(delta, -sun_hat)

    if proj < 0:
        return False  # target is on the sunward side of Moon

    # Perpendicular distance from Moon-Sun axis
    perp = np.linalg.norm(delta + proj * sun_hat)

    return bool(perp < R_MOON)
