"""
orbital_frames.tle — Two-Line Element Set Parser & Propagator
==============================================================

Parses NORAD Two-Line Element sets and propagates satellite state
from the TLE epoch to arbitrary times using analytic J2 perturbations.

The propagator uses mean Keplerian elements with Brouwer J2 secular
rates on RAAN and argument of perigee, plus a simple drag model via
the TLE's Bstar coefficient.  This is *not* a full SGP4 implementation
but is sufficient for task-scheduling visibility windows where ~1 km
position accuracy over hours-to-days is acceptable.

For higher fidelity, feed the TLE-derived epoch state into an external
propagator via the ``epoch_state_eci`` output.

Reference
---------
Vallado, D.A. (2013). *Fundamentals of Astrodynamics*, 4th ed., §9.4.
Hoots, F.R. & Roehrich, R.L. (1980). SPACETRACK Report No. 3.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from .utils import MU_EARTH, R_EARTH, J2, OMEGA_EARTH
from .orbits import (
    keplerian_to_eci, eci_to_keplerian,
    compute_mean_motion, compute_orbital_period,
    _solve_kepler,
)


@dataclass
class TLE:
    """Parsed Two-Line Element set with derived quantities."""
    # ── Raw TLE fields ──
    name: str = ""
    norad_id: int = 0
    classification: str = "U"
    intl_designator: str = ""
    epoch_year: int = 2000
    epoch_day: float = 1.0
    ndot: float = 0.0           # 1st derivative of mean motion [rev/day²]
    nddot: float = 0.0          # 2nd derivative of mean motion [rev/day³]
    bstar: float = 0.0          # B* drag term [1/R_earth]
    element_set: int = 0
    inclination: float = 0.0    # [rad]
    raan: float = 0.0           # [rad]
    eccentricity: float = 0.0
    argp: float = 0.0           # [rad]
    mean_anomaly: float = 0.0   # [rad]
    mean_motion: float = 0.0    # [rad/s]
    rev_number: int = 0

    # ── Derived quantities (computed on parse) ──
    epoch_jd: float = 0.0       # Julian Date of TLE epoch
    semi_major_axis: float = 0.0  # [m]
    period: float = 0.0         # [s]

    # ── User-assigned metadata ──
    priority: int = 5           # scheduling priority (1=highest, 10=lowest)
    revisit_rate: float = 0.0   # desired revisit interval [s]
    catalog_id: str = ""        # user label


def parse_tle(line0: str, line1: str, line2: str) -> TLE:
    """Parse a three-line TLE (name + line 1 + line 2).

    Parameters
    ----------
    line0 : str — satellite name line (may be empty)
    line1 : str — TLE line 1
    line2 : str — TLE line 2

    Returns
    -------
    tle : TLE dataclass
    """
    t = TLE()
    t.name = line0.strip()

    # ── Line 1 ──
    t.norad_id = int(line1[2:7].strip())
    t.classification = line1[7]
    t.intl_designator = line1[9:17].strip()

    # Epoch
    yr = int(line1[18:20].strip())
    t.epoch_year = yr + (1900 if yr >= 57 else 2000)
    t.epoch_day = float(line1[20:32].strip())

    # Mean motion derivatives
    t.ndot = float(line1[33:43].strip())

    # nddot: special format (leading decimal assumed, exponent)
    nddot_str = line1[44:52].strip()
    if nddot_str:
        mantissa = float("0." + nddot_str[:5].replace(" ", "0").replace("+", "").replace("-", ""))
        if nddot_str[0] == '-':
            mantissa = -mantissa
        exp = int(nddot_str[-2:]) if len(nddot_str) > 5 else 0
        t.nddot = mantissa * 10 ** exp
    else:
        t.nddot = 0.0

    # Bstar: same format
    bstar_str = line1[53:61].strip()
    if bstar_str:
        sign = -1.0 if bstar_str[0] == '-' else 1.0
        bstar_clean = bstar_str.lstrip('+-').replace(' ', '0')
        if len(bstar_clean) >= 6:
            mantissa = float("0." + bstar_clean[:5])
            exp = int(bstar_clean[5:].replace('+', '').replace('-', '-') or '0')
            if '-' in bstar_clean[5:]:
                exp = -abs(exp)
            t.bstar = sign * mantissa * 10 ** exp
        else:
            t.bstar = 0.0
    else:
        t.bstar = 0.0

    t.element_set = int(line1[64:68].strip()) if line1[64:68].strip() else 0

    # ── Line 2 ──
    t.inclination = np.deg2rad(float(line2[8:16].strip()))
    t.raan = np.deg2rad(float(line2[17:25].strip()))
    t.eccentricity = float("0." + line2[26:33].strip())
    t.argp = np.deg2rad(float(line2[34:42].strip()))
    t.mean_anomaly = np.deg2rad(float(line2[43:51].strip()))

    # Mean motion [rev/day] → [rad/s]
    mm_rev_day = float(line2[52:63].strip())
    t.mean_motion = mm_rev_day * 2.0 * np.pi / 86400.0

    t.rev_number = int(line2[63:68].strip()) if line2[63:68].strip() else 0

    # ── Derived quantities ──
    t.semi_major_axis = (MU_EARTH / t.mean_motion**2) ** (1.0 / 3.0)
    t.period = compute_orbital_period(t.semi_major_axis)
    t.epoch_jd = _epoch_to_jd(t.epoch_year, t.epoch_day)
    t.catalog_id = f"{t.norad_id:05d}"

    return t


def parse_tle_batch(text: str) -> list[TLE]:
    """Parse multiple TLEs from a multi-line string.

    Handles both 2-line (no name) and 3-line (name + lines) formats.

    Parameters
    ----------
    text : str — concatenated TLE text

    Returns
    -------
    tles : list[TLE]
    """
    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    tles = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            tles.append(parse_tle("", lines[i], lines[i + 1]))
            i += 2
        elif (i + 2 < len(lines) and lines[i + 1].startswith("1 ")
              and lines[i + 2].startswith("2 ")):
            tles.append(parse_tle(lines[i], lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return tles


def _epoch_to_jd(year: int, day_of_year: float) -> float:
    """Convert TLE epoch (year, fractional day) to Julian Date."""
    # Jan 1 of the epoch year
    a = (14 - 1) // 12
    y = year + 4800 - a
    m = 1 + 12 * a - 3
    jd_jan1 = 1 + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd_jan1 -= 0.5  # Julian Date convention (noon)
    return jd_jan1 + day_of_year - 1.0


def tle_epoch_state(tle: TLE) -> tuple[NDArray, NDArray]:
    """Get the ECI state vector at the TLE epoch.

    Uses the mean elements to compute an osculating state.

    Returns
    -------
    r_eci : (3,) — position [m]
    v_eci : (3,) — velocity [m/s]
    """
    # Solve Kepler's equation for true anomaly at epoch
    E = _solve_kepler(tle.mean_anomaly, tle.eccentricity)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + tle.eccentricity) * np.sin(E / 2.0),
        np.sqrt(1.0 - tle.eccentricity) * np.cos(E / 2.0),
    )
    return keplerian_to_eci(
        tle.semi_major_axis, tle.eccentricity, tle.inclination,
        tle.raan, tle.argp, nu,
    )


def propagate_tle(tle: TLE, jd: float) -> tuple[NDArray, NDArray]:
    """Propagate TLE to a given Julian Date using J2 + simple drag.

    Parameters
    ----------
    tle : TLE — parsed TLE
    jd : float — target Julian Date

    Returns
    -------
    r_eci : (3,) — position at jd [m]
    v_eci : (3,) — velocity at jd [m/s]
    """
    dt = (jd - tle.epoch_jd) * 86400.0  # seconds since epoch

    a = tle.semi_major_axis
    e = tle.eccentricity
    inc = tle.inclination
    n = tle.mean_motion
    p = a * (1.0 - e**2)

    # J2 secular rates
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    factor = -1.5 * n * J2 * (R_EARTH / p) ** 2

    raan_dot = factor * cos_i
    argp_dot = factor * (2.0 - 2.5 * sin_i**2)

    # Simple drag: mean motion increases, SMA decreases
    # ndot is in rev/day² from TLE line 1
    n_dot_rad_s2 = tle.ndot * 2.0 * np.pi / 86400.0**2
    n_at_t = n + n_dot_rad_s2 * dt
    a_at_t = (MU_EARTH / n_at_t**2) ** (1.0 / 3.0)

    # Update angular elements
    raan_at_t = tle.raan + raan_dot * dt
    argp_at_t = tle.argp + argp_dot * dt

    # Mean anomaly advance
    M_at_t = (tle.mean_anomaly + n * dt + 0.5 * n_dot_rad_s2 * dt**2) % (2.0 * np.pi)

    # Solve Kepler
    E = _solve_kepler(M_at_t, e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0),
    )

    return keplerian_to_eci(a_at_t, e, inc, raan_at_t, argp_at_t, nu)
