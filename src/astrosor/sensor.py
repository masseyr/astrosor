"""
orbital_frames.sensor — Ground-Based Sensor Models
====================================================

Defines radar and optical ground sensor configurations and computes
satellite visibility including geometric access, elevation constraints,
range limits, field-of-regard, solar illumination conditions, and
visual magnitude estimation for optical sensors.

Sensor Types
------------
- **Radar**: Active illumination — works day/night, needs range/elevation access.
- **Optical**: Passive — requires dark sky, satellite illuminated by Sun,
  solar/lunar exclusion, and target brighter than limiting magnitude.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field as dc_field

from .utils import (
    normalize, R_EARTH, MU_EARTH,
    lla_to_ecef, ecef_to_eci, eci_to_ecef, ecef_to_lla, gmst,
)
from .sun import (
    sun_position_eci, sun_direction_eci, sun_angle,
    is_target_illuminated, eclipse_cylindrical,
    AU,
)
from .moon import moon_direction_eci, moon_angle


# ════════════════════════════════════════════════════════════════════════════
#  Sensor Dataclass
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GroundSensor:
    """Ground-based sensor configuration.

    Parameters
    ----------
    name : str — sensor identifier
    lat : float — geodetic latitude [rad]
    lon : float — geodetic longitude [rad]
    alt : float — altitude above WGS-84 [m]
    sensor_type : str — 'radar' or 'optical'

    Geometric constraints
    ---------------------
    min_elevation : float — minimum elevation angle [rad] (default 10°)
    max_elevation : float — maximum elevation angle [rad] (default 90°)
    min_azimuth : float — minimum azimuth [rad] (default 0)
    max_azimuth : float — maximum azimuth [rad] (default 2π, full)
    min_range : float — minimum slant range [m]
    max_range : float — maximum slant range [m]
    field_of_regard : float — half-angle of full FOR cone [rad] (overrides az/el if set)

    Optical sensor parameters
    -------------------------
    limiting_magnitude : float — faintest detectable visual magnitude
    solar_exclusion_angle : float — min boresight-to-Sun angle [rad]
    lunar_exclusion_angle : float — min boresight-to-Moon angle [rad]
    max_sun_elevation : float — Sun must be below this elevation for dark sky [rad]
        (negative = below horizon; default -12° = nautical twilight)

    Radar parameters
    ----------------
    sensitivity_area : float — effective aperture × gain factor [m²]
        (used for SNR-like range estimation; not a full radar equation)
    """
    name: str = "Sensor"
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    sensor_type: str = "optical"  # 'radar' or 'optical'

    # Geometric
    min_elevation: float = np.deg2rad(10.0)
    max_elevation: float = np.deg2rad(90.0)
    min_azimuth: float = 0.0
    max_azimuth: float = 2.0 * np.pi
    min_range: float = 0.0
    max_range: float = np.inf
    field_of_regard: float = 0.0  # 0 = use az/el limits instead

    # Optical
    limiting_magnitude: float = 16.0
    solar_exclusion_angle: float = np.deg2rad(30.0)
    lunar_exclusion_angle: float = np.deg2rad(10.0)
    max_sun_elevation: float = np.deg2rad(-12.0)  # nautical twilight

    # Radar
    sensitivity_area: float = 100.0

    def ecef(self) -> NDArray:
        """Sensor position in ECEF [m]."""
        return lla_to_ecef(self.lat, self.lon, self.alt)

    def eci(self, jd: float) -> NDArray:
        """Sensor position in ECI at given Julian Date [m]."""
        return ecef_to_eci(self.ecef(), jd)

    def eci_velocity(self, jd: float) -> NDArray:
        """Sensor velocity in ECI (Earth rotation) [m/s]."""
        r_eci = self.eci(jd)
        omega = np.array([0.0, 0.0, 7.2921150e-5])
        return np.cross(omega, r_eci)


# ════════════════════════════════════════════════════════════════════════════
#  Topocentric Geometry
# ════════════════════════════════════════════════════════════════════════════

def topocentric_azel(
    r_sensor_ecef: NDArray, lat: float, lon: float,
    r_target_ecef: NDArray,
) -> tuple[float, float, float]:
    """Compute azimuth, elevation, and range from sensor to target.

    Parameters
    ----------
    r_sensor_ecef : (3,) — sensor ECEF position [m]
    lat, lon : float — sensor geodetic latitude/longitude [rad]
    r_target_ecef : (3,) — target ECEF position [m]

    Returns
    -------
    az : float — azimuth [rad], 0=North, π/2=East
    el : float — elevation [rad], 0=horizon, π/2=zenith
    rng : float — slant range [m]
    """
    delta = r_target_ecef - r_sensor_ecef
    rng = np.linalg.norm(delta)

    # SEZ (South-East-Zenith) rotation
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # ECEF → SEZ
    S = (-sin_lat * cos_lon * delta[0]
         - sin_lat * sin_lon * delta[1]
         + cos_lat * delta[2])
    E = (-sin_lon * delta[0] + cos_lon * delta[1])
    Z = (cos_lat * cos_lon * delta[0]
         + cos_lat * sin_lon * delta[1]
         + sin_lat * delta[2])

    el = np.arctan2(Z, np.sqrt(S**2 + E**2))
    az = np.arctan2(E, -S) % (2.0 * np.pi)  # -S because S points South

    return float(az), float(el), float(rng)


def sun_elevation_at_sensor(sensor: GroundSensor, jd: float) -> float:
    """Compute Sun elevation as seen from the sensor [rad].

    Negative = below horizon.
    """
    r_sun_eci = sun_position_eci(jd)
    r_sun_ecef = eci_to_ecef(r_sun_eci, jd)
    _, el, _ = topocentric_azel(sensor.ecef(), sensor.lat, sensor.lon, r_sun_ecef)
    return el


# ════════════════════════════════════════════════════════════════════════════
#  Visual Magnitude Estimation
# ════════════════════════════════════════════════════════════════════════════

def estimate_visual_magnitude(
    r_sat_eci: NDArray, r_sensor_eci: NDArray, jd: float,
    intrinsic_magnitude: float = 6.0,
    rcs: float = 1.0,
) -> float:
    """Estimate visual magnitude of a satellite as seen from a ground sensor.

    Uses a simplified magnitude model based on range and solar phase angle.
    The model assumes diffuse (Lambertian) reflection from a sphere of
    equivalent radar cross-section ``rcs``.

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    r_sensor_eci : (3,) — sensor position in ECI [m]
    jd : float — Julian Date
    intrinsic_magnitude : float — reference magnitude at 1000 km range,
        zero phase angle, and 1 m² RCS.  Default 6.0 (typical LEO).
    rcs : float — radar cross-section / effective reflective area [m²]

    Returns
    -------
    vmag : float — estimated visual magnitude (brighter = more negative)
    """
    r_sat = np.asarray(r_sat_eci, dtype=np.float64)
    r_sens = np.asarray(r_sensor_eci, dtype=np.float64)
    r_sun = sun_position_eci(jd)

    # Slant range
    delta = r_sat - r_sens
    slant_range = np.linalg.norm(delta)

    # Phase angle: Sun-satellite-observer
    to_sun = normalize(r_sun - r_sat)
    to_obs = normalize(r_sens - r_sat)
    cos_phase = np.clip(np.dot(to_sun, to_obs), -1.0, 1.0)
    phase = np.arccos(cos_phase)

    # Lambertian phase function: (1 + cos(phase)) / 2
    # Normalized so full illumination (phase=0) → 1.0
    phase_func = (1.0 + cos_phase) / 2.0
    phase_func = max(phase_func, 1e-6)  # prevent log(0)

    # Range factor: magnitude increases with 5*log10(range/1000km)
    range_ref = 1e6  # 1000 km reference
    range_factor = 5.0 * np.log10(slant_range / range_ref)

    # RCS factor: brighter with larger cross-section
    rcs_factor = -2.5 * np.log10(max(rcs, 0.01))

    # Phase factor
    phase_factor = -2.5 * np.log10(phase_func)

    vmag = intrinsic_magnitude + range_factor + rcs_factor + phase_factor

    return float(vmag)


# ════════════════════════════════════════════════════════════════════════════
#  Visibility Check
# ════════════════════════════════════════════════════════════════════════════

def check_visibility(
    sensor: GroundSensor,
    r_sat_eci: NDArray,
    jd: float,
    rcs: float = 1.0,
    intrinsic_mag: float = 6.0,
) -> dict:
    """Comprehensive single-epoch visibility check.

    Evaluates all geometric and environmental constraints for both
    radar and optical sensors.

    Parameters
    ----------
    sensor : GroundSensor
    r_sat_eci : (3,) — satellite ECI position [m]
    jd : float — Julian Date
    rcs : float — satellite RCS [m²] (for optical mag estimation)
    intrinsic_mag : float — reference magnitude

    Returns
    -------
    dict with:
        'visible' : bool — passes all constraints
        'az' : float — azimuth [rad]
        'el' : float — elevation [rad]
        'range' : float — slant range [m]
        'in_elevation' : bool
        'in_azimuth' : bool
        'in_range' : bool
        'in_for' : bool — within field of regard (if defined)
        'reasons' : list[str] — why NOT visible (empty if visible)

        For optical sensors additionally:
        'sky_dark' : bool — Sun below max_sun_elevation
        'sat_illuminated' : bool — satellite in sunlight
        'sun_clear' : bool — outside solar exclusion
        'moon_clear' : bool — outside lunar exclusion
        'visual_mag' : float — estimated magnitude
        'mag_detectable' : bool — brighter than limiting mag
    """
    r_sat = np.asarray(r_sat_eci, dtype=np.float64)
    r_sensor_ecef = sensor.ecef()
    r_sat_ecef = eci_to_ecef(r_sat, jd)
    r_sensor_eci = sensor.eci(jd)

    az, el, rng = topocentric_azel(r_sensor_ecef, sensor.lat, sensor.lon, r_sat_ecef)

    reasons = []
    result = {
        "az": az, "el": el, "range": rng,
    }

    # ── Elevation ──
    in_el = sensor.min_elevation <= el <= sensor.max_elevation
    result["in_elevation"] = in_el
    if not in_el:
        reasons.append("elevation")

    # ── Azimuth ──
    if sensor.min_azimuth == 0.0 and sensor.max_azimuth >= 2.0 * np.pi - 0.01:
        in_az = True
    else:
        # Handle wrap-around
        if sensor.min_azimuth <= sensor.max_azimuth:
            in_az = sensor.min_azimuth <= az <= sensor.max_azimuth
        else:
            in_az = az >= sensor.min_azimuth or az <= sensor.max_azimuth
    result["in_azimuth"] = in_az
    if not in_az:
        reasons.append("azimuth")

    # ── Range ──
    in_rng = sensor.min_range <= rng <= sensor.max_range
    result["in_range"] = in_rng
    if not in_rng:
        reasons.append("range")

    # ── Field of Regard (conical) ──
    if sensor.field_of_regard > 0:
        zenith_angle = np.pi / 2.0 - el
        in_for = zenith_angle <= sensor.field_of_regard
        result["in_for"] = in_for
        if not in_for:
            reasons.append("field_of_regard")
    else:
        result["in_for"] = True  # no FOR limit → always in

    # ── Radar: geometric only ──
    if sensor.sensor_type == "radar":
        result["visible"] = len(reasons) == 0
        result["reasons"] = reasons
        return result

    # ── Optical-specific checks ──

    # Sky darkness
    sun_el = sun_elevation_at_sensor(sensor, jd)
    sky_dark = sun_el <= sensor.max_sun_elevation
    result["sky_dark"] = sky_dark
    result["sun_elevation"] = float(sun_el)
    if not sky_dark:
        reasons.append("sky_bright")

    # Satellite illumination (must be in sunlight)
    sat_lit = is_target_illuminated(r_sat, jd)
    result["sat_illuminated"] = sat_lit
    if not sat_lit:
        reasons.append("sat_in_shadow")

    # Solar exclusion
    los_dir = normalize(r_sat - r_sensor_eci)
    s_angle = sun_angle(r_sensor_eci, los_dir, jd)
    sun_clear = s_angle >= sensor.solar_exclusion_angle
    result["sun_clear"] = sun_clear
    result["sun_angle"] = float(s_angle)
    if not sun_clear:
        reasons.append("solar_exclusion")

    # Lunar exclusion
    m_angle = moon_angle(r_sensor_eci, los_dir, jd)
    moon_clear = m_angle >= sensor.lunar_exclusion_angle
    result["moon_clear"] = moon_clear
    result["moon_angle"] = float(m_angle)
    if not moon_clear:
        reasons.append("lunar_exclusion")

    # Visual magnitude
    vmag = estimate_visual_magnitude(r_sat, r_sensor_eci, jd, intrinsic_mag, rcs)
    mag_ok = vmag <= sensor.limiting_magnitude
    result["visual_mag"] = vmag
    result["mag_detectable"] = mag_ok
    if not mag_ok:
        reasons.append("too_dim")

    result["visible"] = len(reasons) == 0
    result["reasons"] = reasons
    return result
