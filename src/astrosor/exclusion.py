"""
orbital_frames.exclusion — Combined Solar & Lunar Exclusion Analysis
=====================================================================

Integrates sun and moon modules with the coverage framework to provide
a unified sensor availability timeline accounting for:
  - Solar exclusion zone (boresight keep-out)
  - Lunar exclusion zone (boresight keep-out)
  - Earth shadow / eclipse (target or sensor not illuminated)
  - Earth limb exclusion (boresight too close to Earth's limb)

The primary output is a time-tagged availability mask that can be
combined with coverage windows to compute effective sensor duty cycle.
"""

import numpy as np
from numpy.typing import NDArray

from .utils import normalize, R_EARTH
from .sun import (
    sun_angle, sun_direction_eci, eclipse_conical, eclipse_cylindrical,
    solar_phase_angle, is_target_illuminated,
)
from .moon import (
    moon_angle, moon_direction_eci, moon_illumination_fraction,
)


# ════════════════════════════════════════════════════════════════════════════
#  Combined Exclusion Check (Single Epoch)
# ════════════════════════════════════════════════════════════════════════════

def check_all_exclusions(
    r_sat_eci: NDArray,
    v_sat_eci: NDArray,
    boresight_eci: NDArray,
    jd: float,
    solar_exclusion_half_angle: float = np.deg2rad(30.0),
    lunar_exclusion_half_angle: float = np.deg2rad(10.0),
    earth_limb_half_angle: float = np.deg2rad(5.0),
    require_sunlit_sensor: bool = False,
) -> dict:
    """Evaluate all exclusion constraints at a single epoch.

    Parameters
    ----------
    r_sat_eci : (3,) — satellite position in ECI [m]
    v_sat_eci : (3,) — satellite velocity in ECI [m/s]
    boresight_eci : (3,) — sensor boresight unit vector in ECI
    jd : float — Julian Date
    solar_exclusion_half_angle : float — Sun keep-out [rad] (default 30°)
    lunar_exclusion_half_angle : float — Moon keep-out [rad] (default 10°)
    earth_limb_half_angle : float — Earth limb keep-out [rad] (default 5°)
    require_sunlit_sensor : bool — if True, sensor must be in sunlight

    Returns
    -------
    dict with:
        'available' : bool — True if NO exclusions are violated
        'sun_excluded' : bool — boresight in solar keep-out zone
        'moon_excluded' : bool — boresight in lunar keep-out zone
        'earth_excluded' : bool — boresight too close to Earth limb
        'eclipse' : bool — sensor is in Earth's shadow
        'sun_angle' : float — boresight-to-Sun angle [rad]
        'moon_angle' : float — boresight-to-Moon angle [rad]
        'earth_angle' : float — boresight-to-Earth-limb angle [rad]
        'eclipse_state' : str — 'sunlit', 'penumbra', or 'umbra'
        'violations' : list[str] — names of active exclusions
    """
    r = np.asarray(r_sat_eci, dtype=np.float64)
    b = normalize(np.asarray(boresight_eci, dtype=np.float64))

    violations = []

    # ── Solar exclusion ──
    s_angle = sun_angle(r, b, jd)
    sun_exc = bool(s_angle < solar_exclusion_half_angle)
    if sun_exc:
        violations.append("solar")

    # ── Lunar exclusion ──
    m_angle = moon_angle(r, b, jd)
    moon_exc = bool(m_angle < lunar_exclusion_half_angle)
    if moon_exc:
        violations.append("lunar")

    # ── Earth limb exclusion ──
    # Angular radius of Earth as seen from the satellite
    r_mag = np.linalg.norm(r)
    earth_angular_radius = np.arcsin(np.clip(R_EARTH / r_mag, 0.0, 1.0))
    # Angle from boresight to nadir (Earth center)
    nadir = -normalize(r)
    bore_nadir_angle = np.arccos(np.clip(np.dot(b, nadir), -1.0, 1.0))
    # Angle from boresight to Earth limb
    earth_limb_angle = bore_nadir_angle - earth_angular_radius
    earth_exc = bool(earth_limb_angle < earth_limb_half_angle
                     and bore_nadir_angle < np.pi / 2 + earth_angular_radius)
    # Only flag if boresight is actually pointing toward Earth's disk region
    if earth_exc and bore_nadir_angle < earth_angular_radius + earth_limb_half_angle:
        violations.append("earth_limb")
    else:
        earth_exc = False

    # ── Eclipse ──
    ecl = eclipse_conical(r, jd)
    in_eclipse = ecl["state"] != "sunlit"
    if require_sunlit_sensor and in_eclipse:
        violations.append("eclipse")

    available = len(violations) == 0

    return {
        "available": available,
        "sun_excluded": sun_exc,
        "moon_excluded": moon_exc,
        "earth_excluded": earth_exc,
        "eclipse": in_eclipse,
        "sun_angle": float(s_angle),
        "moon_angle": float(m_angle),
        "earth_angle": float(earth_limb_angle),
        "eclipse_state": ecl["state"],
        "violations": violations,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Availability Timeline
# ════════════════════════════════════════════════════════════════════════════

def availability_timeline(
    r0_eci: NDArray, v0_eci: NDArray,
    boresight_body_fn,
    duration: float,
    dt_step: float = 30.0,
    jd_epoch: float = 2451545.0,
    solar_exclusion_half_angle: float = np.deg2rad(30.0),
    lunar_exclusion_half_angle: float = np.deg2rad(10.0),
    earth_limb_half_angle: float = np.deg2rad(5.0),
    require_sunlit_sensor: bool = False,
    propagate_fn=None,
) -> dict:
    """Generate a time-tagged sensor availability timeline.

    Parameters
    ----------
    r0_eci, v0_eci : (3,) — initial satellite ECI state
    boresight_body_fn : callable(r_eci, v_eci) → (3,) ndarray
        Returns boresight unit vector in ECI.
    duration : float — analysis span [s]
    dt_step : float — time step [s]
    jd_epoch : float — Julian Date at t=0
    solar_exclusion_half_angle : float [rad]
    lunar_exclusion_half_angle : float [rad]
    earth_limb_half_angle : float [rad]
    require_sunlit_sensor : bool
    propagate_fn : callable or None

    Returns
    -------
    dict with:
        'times' : (N,) ndarray — time stamps [s from epoch]
        'available' : (N,) bool array — sensor available at each step
        'sun_excluded' : (N,) bool array
        'moon_excluded' : (N,) bool array
        'earth_excluded' : (N,) bool array
        'eclipse' : (N,) bool array
        'sun_angles' : (N,) float array [rad]
        'moon_angles' : (N,) float array [rad]
        'duty_cycle' : float — fraction of time available [0..1]
        'exclusion_summary' : dict — breakdown of exclusion causes
    """
    from .orbits import propagate_kepler
    prop = propagate_fn or propagate_kepler

    times = np.arange(0, duration, dt_step)
    N = len(times)

    available = np.ones(N, dtype=bool)
    sun_exc = np.zeros(N, dtype=bool)
    moon_exc = np.zeros(N, dtype=bool)
    earth_exc = np.zeros(N, dtype=bool)
    eclipse = np.zeros(N, dtype=bool)
    sun_angles = np.zeros(N)
    moon_angles = np.zeros(N)

    for k, t in enumerate(times):
        r, v = prop(r0_eci, v0_eci, t)
        jd = jd_epoch + t / 86400.0
        bore = boresight_body_fn(r, v)

        result = check_all_exclusions(
            r, v, bore, jd,
            solar_exclusion_half_angle=solar_exclusion_half_angle,
            lunar_exclusion_half_angle=lunar_exclusion_half_angle,
            earth_limb_half_angle=earth_limb_half_angle,
            require_sunlit_sensor=require_sunlit_sensor,
        )

        available[k] = result["available"]
        sun_exc[k] = result["sun_excluded"]
        moon_exc[k] = result["moon_excluded"]
        earth_exc[k] = result["earth_excluded"]
        eclipse[k] = result["eclipse"]
        sun_angles[k] = result["sun_angle"]
        moon_angles[k] = result["moon_angle"]

    duty_cycle = float(np.mean(available))

    summary = {
        "solar_exclusion_fraction": float(np.mean(sun_exc)),
        "lunar_exclusion_fraction": float(np.mean(moon_exc)),
        "earth_limb_fraction": float(np.mean(earth_exc)),
        "eclipse_fraction": float(np.mean(eclipse)),
        "total_unavailable_fraction": 1.0 - duty_cycle,
    }

    return {
        "times": times,
        "available": available,
        "sun_excluded": sun_exc,
        "moon_excluded": moon_exc,
        "earth_excluded": earth_exc,
        "eclipse": eclipse,
        "sun_angles": sun_angles,
        "moon_angles": moon_angles,
        "duty_cycle": duty_cycle,
        "exclusion_summary": summary,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Target Observability
# ════════════════════════════════════════════════════════════════════════════

def target_observability(
    r_sat_eci: NDArray, v_sat_eci: NDArray,
    r_target_eci: NDArray,
    jd: float,
    sensor_half_cone: float,
    solar_exclusion_half_angle: float = np.deg2rad(30.0),
    lunar_exclusion_half_angle: float = np.deg2rad(10.0),
    require_illuminated_target: bool = True,
) -> dict:
    """Full observability assessment for a specific target at one epoch.

    Combines geometric access, solar/lunar exclusion, eclipse, and
    target illumination into a single pass/fail with diagnostics.

    Parameters
    ----------
    r_sat_eci, v_sat_eci : (3,) — sensor satellite ECI state
    r_target_eci : (3,) — target position in ECI [m]
    jd : float — Julian Date
    sensor_half_cone : float — sensor FOV half-cone angle [rad]
    solar_exclusion_half_angle : float [rad]
    lunar_exclusion_half_angle : float [rad]
    require_illuminated_target : bool — target must be sunlit

    Returns
    -------
    dict with:
        'observable' : bool — True if target is observable
        'in_fov' : bool — target within sensor FOV
        'range' : float — slant range [m]
        'sun_clear' : bool — boresight outside solar exclusion
        'moon_clear' : bool — boresight outside lunar exclusion
        'sensor_sunlit' : bool — sensor not eclipsed
        'target_illuminated' : bool — target in sunlight
        'phase_angle' : float — Sun-Target-Sensor angle [rad]
        'reasons' : list[str] — reasons for non-observability
    """
    r_s = np.asarray(r_sat_eci, dtype=np.float64)
    r_t = np.asarray(r_target_eci, dtype=np.float64)

    delta = r_t - r_s
    rng = np.linalg.norm(delta)
    los = delta / rng  # line-of-sight unit vector

    reasons = []

    # ── FOV check ──
    # For nadir-pointing, the boresight is -r_hat, but we check LOS directly
    # against sensor half-cone from boresight.  Here we assume boresight = -r_hat.
    boresight = -normalize(r_s)
    off_bore = np.arccos(np.clip(np.dot(los, boresight), -1.0, 1.0))
    in_fov = bool(off_bore <= sensor_half_cone)
    if not in_fov:
        reasons.append("out_of_fov")

    # ── Solar exclusion ──
    s_ang = sun_angle(r_s, los, jd)
    sun_clear = bool(s_ang >= solar_exclusion_half_angle)
    if not sun_clear:
        reasons.append("solar_exclusion")

    # ── Lunar exclusion ──
    m_ang = moon_angle(r_s, los, jd)
    moon_clear = bool(m_ang >= lunar_exclusion_half_angle)
    if not moon_clear:
        reasons.append("lunar_exclusion")

    # ── Sensor eclipse ──
    ecl = eclipse_conical(r_s, jd)
    sensor_sunlit = ecl["state"] == "sunlit"

    # ── Target illumination ──
    target_lit = is_target_illuminated(r_t, jd)
    if require_illuminated_target and not target_lit:
        reasons.append("target_dark")

    # ── Phase angle ──
    phase = solar_phase_angle(r_s, r_t, jd)

    observable = in_fov and sun_clear and moon_clear
    if require_illuminated_target:
        observable = observable and target_lit

    return {
        "observable": observable,
        "in_fov": in_fov,
        "range": float(rng),
        "sun_clear": sun_clear,
        "moon_clear": moon_clear,
        "sensor_sunlit": sensor_sunlit,
        "target_illuminated": target_lit,
        "phase_angle": float(phase),
        "sun_angle": float(s_ang),
        "moon_angle": float(m_ang),
        "reasons": reasons,
    }
