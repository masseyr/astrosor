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
from datetime import datetime, timedelta
from dataclasses import dataclass, field


from .utils import MU_EARTH, R_EARTH, J2, OMEGA_EARTH, DAILY_SECONDS, SQRT_MU
from .orbits import (
    keplerian_to_eci, eci_to_keplerian,
    compute_mean_motion, compute_orbital_period,
    _solve_kepler,
)



def format_as_bald_decimal_str(val:float,
                               use_plus='+') \
                               -> str:
    _str = f"{abs(val):.8f}".removeprefix("0.")
    sign = use_plus if val > 0 else "-"
    return "".join([sign, ".", _str])

def format_as_tle_exp_str(val:float,
                          precision=8,
                          length=8,
                          use_plus=' ') \
                          -> str:
    mantissa_str, exponent_str = ("{:.{}e}".format(val, precision)).split('e')
    exponent_sign = "+" if int(exponent_str) > 0 else "-"
    sign = use_plus if val > 0 else "-"

    exponent_str = str(abs(int(exponent_str)))
    len_mantissa_str = length - len(sign) - len(exponent_str)
    mantissa_str = mantissa_str.replace("e", "").replace(".", "")[:len_mantissa_str]

    return "".join([sign, mantissa_str, exponent_sign, exponent_str])


def mean_motion_to_semi_maj_axis(mean_motion):
    if mean_motion != 0.0:
        orbital_period = DAILY_SECONDS/mean_motion
        return np.pow(orbital_period*SQRT_MU/(2.0*np.pi), (2.0/3.0))
    return 0.0

def semi_maj_axis_to_mean_motion(semi_maj_axis):
    orbital_period = (2.0*np.pi)* np.sqrt((semi_maj_axis**3) / (MU_EARTH / 1e9))
    return DAILY_SECONDS/orbital_period


def calc_checksum(tle_line_without_checksum:str):
    checksum = 0
    for char in tle_line_without_checksum:
        if char.isdigit():
            checksum += int(char)
        elif char == "-":
            checksum += 1
    return checksum % 10



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

    @property
    def period(self):
        """Full orbit period in minutes"""
        return (24. * 60.)/self.mean_motion
    
    @property
    def apogee(self):
        if self.semi_major_axis > 0.:
            return (self.semi_major_axis * (1 + self.eccentricity)) - (R_EARTH/1000.)

    @property
    def perigee(self):
        if self.semi_major_axis > 0.:
            return (self.semi_major_axis * (1 - self.eccentricity)) - (R_EARTH/1000.)

    def orbit_type(self) -> str:
        """
        Returns obit type string: LEO, HEO, MEO or GEO
        """
        if self.period < 225:
            return "LEO"
        elif self.eccentricity >= 0.3:
            return "HEO"
        elif self.period < 800:
            return "MEO"
        else:
            return "GEO"
        

def parse_tle(*lines) -> TLE:
    """Parse a three-line TLE (name + line 1 + line 2).

    Parameters
    ----------
    lines: TLE lines

    Returns
    -------
    tle : TLE dataclass
    """
    line0 = ''
    if len(lines) == 3:
        line0, line1, line2 = lines
    elif len(lines) == 2:
        line1, line2 = lines

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
    jd_jan1 = 1 + ((153 * m + 2) // 5) + 365 * y + (y // 4) - (y // 100) + (y // 400) - 32045
    jd_jan1 -= 0.5  # Julian Date convention (noon)
    return jd_jan1 + day_of_year 


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


# ════════════════════════════════════════════════════════════════════════════
#  TLE Checksum
# ════════════════════════════════════════════════════════════════════════════

def tle_checksum(line: str) -> int:
    """Compute the modulo-10 checksum for a TLE line.

    Each digit contributes its face value, '-' counts as 1,
    all other characters count as 0.  The result is sum mod 10.

    Parameters
    ----------
    line : str — a TLE line (characters 0..67; column 69 is the checksum)

    Returns
    -------
    checksum : int — single digit 0-9
    """
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == '-':
            s += 1
    return s % 10


def verify_checksum(line: str) -> bool:
    """Verify the modulo-10 checksum of a TLE line.

    Parameters
    ----------
    line : str — a complete TLE line (69 characters with checksum at column 69)

    Returns
    -------
    valid : bool — True if the last digit matches the computed checksum
    """
    if len(line) < 69:
        return False
    expected = int(line[68])
    return tle_checksum(line) == expected


def verify_tle(line1: str, line2: str) -> dict:
    """Verify checksums and basic structural validity of a TLE pair.

    Parameters
    ----------
    line1, line2 : str — TLE lines 1 and 2

    Returns
    -------
    dict with:
        'valid' : bool — both lines pass all checks
        'line1_checksum' : bool — line 1 checksum valid
        'line2_checksum' : bool — line 2 checksum valid
        'line1_prefix' : bool — line 1 starts with '1 '
        'line2_prefix' : bool — line 2 starts with '2 '
        'norad_match' : bool — NORAD IDs match between lines
        'errors' : list[str]
    """
    errors = []

    l1_pfx = line1.startswith("1 ")
    l2_pfx = line2.startswith("2 ")
    if not l1_pfx:
        errors.append("line1 does not start with '1 '")
    if not l2_pfx:
        errors.append("line2 does not start with '2 '")

    l1_ck = verify_checksum(line1)
    l2_ck = verify_checksum(line2)
    if not l1_ck:
        errors.append(f"line1 checksum: expected {tle_checksum(line1)}, got {line1[68] if len(line1) >= 69 else '?'}")
    if not l2_ck:
        errors.append(f"line2 checksum: expected {tle_checksum(line2)}, got {line2[68] if len(line2) >= 69 else '?'}")

    # NORAD ID match
    try:
        id1 = int(line1[2:7].strip())
        id2 = int(line2[2:7].strip())
        norad_ok = id1 == id2
        if not norad_ok:
            errors.append(f"NORAD ID mismatch: line1={id1}, line2={id2}")
    except (ValueError, IndexError):
        norad_ok = False
        errors.append("could not parse NORAD IDs")

    return {
        "valid": l1_pfx and l2_pfx and l1_ck and l2_ck and norad_ok,
        "line1_checksum": l1_ck,
        "line2_checksum": l2_ck,
        "line1_prefix": l1_pfx,
        "line2_prefix": l2_pfx,
        "norad_match": norad_ok,
        "errors": errors,
    }


# ════════════════════════════════════════════════════════════════════════════
#  TLE Export (Format to Strings)
# ════════════════════════════════════════════════════════════════════════════

def _format_exp_field(value: float) -> str:
    """Format a value in TLE's special exponent notation.

    TLE format: ±NNNNN±E  where value = ±0.NNNNN × 10^±E
    Example: 0.000123 → ' 12300-3'  (note: leading space or minus)
             -0.00456 → '-45600-2'
    """
    if value == 0.0:
        return " 00000-0"

    sign = '-' if value < 0 else ' '
    val = abs(value)

    # Find exponent so that 0.1 <= mantissa < 1.0
    exp = 0
    mantissa = val
    if mantissa != 0:
        exp = int(np.floor(np.log10(mantissa))) + 1
        mantissa = val / (10.0 ** exp)

    # mantissa is now in [0.1, 1.0)
    digits = f"{mantissa:.5f}"[2:7]  # 5 digits after '0.'

    exp_sign = '+' if exp >= 0 else '-'
    return f"{sign}{digits}{exp_sign}{abs(exp)}"


def tle_to_lines(tle: TLE) -> tuple[str, str]:
    """Export a TLE dataclass back to standard two-line element strings.

    Produces properly formatted 69-character lines with valid checksums.

    Parameters
    ----------
    tle : TLE — parsed or constructed TLE

    Returns
    -------
    line1, line2 : str — TLE line 1 and line 2 (69 chars each)
    """
    # ── Epoch ──
    yr_2d = tle.epoch_year % 100
    epoch_str = f"{yr_2d:02d}{tle.epoch_day:012.8f}"

    # ── ndot ──
    # ndot is in rev/day², formatted as ±.NNNNNNNN (leading decimal assumed)
    ndot_str = f"{tle.ndot:+011.8f}".replace("+", " ")
    # TLE columns 34-43 (10 chars): " .NNNNNNNN" or "-.NNNNNNNN"
    if tle.ndot >= 0:
        ndot_field = f" {abs(tle.ndot):.8f}"[:10]
    else:
        ndot_field = f"{tle.ndot:.8f}"[:10]

    # ── nddot and bstar: special exponent format ──
    nddot_field = _format_exp_field(tle.nddot)
    bstar_field = _format_exp_field(tle.bstar)

    # ── International designator ──
    intl = f"{tle.intl_designator:<8s}"

    # ── Element set type (always 0 for SGP4) ──
    etype = 0

    # ── Element set number ──
    elset = f"{tle.element_set:4d}"

    # ── Build Line 1 (without checksum) ──
    line1_body = (
        f"1 {tle.norad_id:05d}{tle.classification} "
        f"{intl} "
        f"{epoch_str} "
        f"{ndot_field} "
        f"{nddot_field} "
        f"{bstar_field} "
        f"{etype}"
        f"{elset}"
    )
    # Pad or trim to exactly 68 characters
    line1_body = f"{line1_body:<68s}"[:68]
    line1 = line1_body + str(tle_checksum(line1_body))

    # ── Line 2 ──
    inc_deg = np.rad2deg(tle.inclination) % 360
    raan_deg = np.rad2deg(tle.raan) % 360
    argp_deg = np.rad2deg(tle.argp) % 360
    ma_deg = np.rad2deg(tle.mean_anomaly) % 360
    ecc_str = f"{tle.eccentricity:.7f}"[2:]  # drop "0."
    mm_rev_day = tle.mean_motion * 86400.0 / (2.0 * np.pi)
    rev_num = tle.rev_number % 100000

    line2_body = (
        f"2 {tle.norad_id:05d} "
        f"{inc_deg:8.4f} "
        f"{raan_deg:8.4f} "
        f"{ecc_str} "
        f"{argp_deg:8.4f} "
        f"{ma_deg:8.4f} "
        f"{mm_rev_day:11.8f}"
        f"{rev_num:5d}"
    )
    line2_body = f"{line2_body:<68s}"[:68]
    line2 = line2_body + str(tle_checksum(line2_body))

    return line1, line2


def tle_to_string(tle: TLE, include_name: bool = True) -> str:
    """Export a TLE to a complete multi-line string.

    Parameters
    ----------
    tle : TLE
    include_name : bool — if True, prepend the satellite name line

    Returns
    -------
    text : str — 2 or 3 line TLE string
    """
    line1, line2 = tle_to_lines(tle)
    if include_name and tle.name:
        return f"{tle.name}\n{line1}\n{line2}"
    return f"{line1}\n{line2}"


# ════════════════════════════════════════════════════════════════════════════
#  TLE Epoch & Mean Anomaly Update
# ════════════════════════════════════════════════════════════════════════════

def _jd_to_epoch(jd: float) -> tuple[int, float]:
    """Convert Julian Date to TLE epoch (year, fractional day-of-year)."""
    # Julian Date → calendar date (Meeus algorithm)
    z = int(jd + 0.5)
    f = (jd + 0.5) - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    # Day of year
    # Jan 1 JD for this year
    jd_jan1 = _epoch_to_jd(year, 1.0)
    day_of_year = jd - jd_jan1 + 1.0

    return int(year), float(day_of_year)


def update_epoch(tle: TLE, new_jd: float, propagate: bool = True) -> TLE:
    """Create a new TLE with an updated epoch and propagated mean anomaly.

    Advances (or retards) the mean anomaly by the time difference
    from the old epoch to the new epoch, applying J2 secular rates
    on RAAN and argument of perigee and drag on mean motion.

    The returned TLE is a new object; the original is not modified.

    Parameters
    ----------
    tle : TLE — source TLE
    new_jd : float — new epoch Julian Date
    propagate : bool — if True, propagate mean anomaly, RAAN, argp,
        and mean motion to the new epoch.  If False, only update
        the epoch timestamp (mean anomaly unchanged).

    Returns
    -------
    new_tle : TLE — updated copy
    """
    import copy
    t = copy.deepcopy(tle)

    new_year, new_day = _jd_to_epoch(new_jd)
    t.epoch_year = new_year
    t.epoch_day = new_day
    t.epoch_jd = new_jd

    if propagate:
        dt = (new_jd - tle.epoch_jd) * 86400.0  # seconds

        n = tle.mean_motion
        e = tle.eccentricity
        a = tle.semi_major_axis
        p = a * (1.0 - e**2)

        # J2 secular rates
        cos_i = np.cos(tle.inclination)
        sin_i = np.sin(tle.inclination)
        factor = -1.5 * n * J2 * (R_EARTH / p) ** 2

        raan_dot = factor * cos_i
        argp_dot = factor * (2.0 - 2.5 * sin_i**2)

        # Drag: ndot in rev/day²
        n_dot_rad_s2 = tle.ndot * 2.0 * np.pi / 86400.0**2

        # Update elements
        t.raan = (tle.raan + raan_dot * dt) % (2.0 * np.pi)
        t.argp = (tle.argp + argp_dot * dt) % (2.0 * np.pi)
        t.mean_anomaly = (tle.mean_anomaly + n * dt
                          + 0.5 * n_dot_rad_s2 * dt**2) % (2.0 * np.pi)

        # Updated mean motion (drag)
        t.mean_motion = n + n_dot_rad_s2 * dt
        t.semi_major_axis = (MU_EARTH / t.mean_motion**2) ** (1.0 / 3.0)
        t.period = compute_orbital_period(t.semi_major_axis)

        # Advance rev number by approximate number of orbits
        revs = abs(dt) / tle.period
        if dt >= 0:
            t.rev_number = tle.rev_number + int(revs)
        else:
            t.rev_number = max(0, tle.rev_number - int(revs))

    return t


def update_mean_anomaly(tle: TLE, new_mean_anomaly: float) -> TLE:
    """Create a new TLE with a replaced mean anomaly (no propagation).

    Parameters
    ----------
    tle : TLE — source TLE
    new_mean_anomaly : float — new mean anomaly [rad]

    Returns
    -------
    new_tle : TLE — copy with updated mean anomaly
    """
    import copy
    t = copy.deepcopy(tle)
    t.mean_anomaly = new_mean_anomaly % (2.0 * np.pi)
    return t
