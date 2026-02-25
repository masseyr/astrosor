"""
orbital_frames — ECI / ECR / UNW / TNW Coordinate Frame Library
================================================================

A pure-NumPy library for transforming between Earth-Centered Inertial (ECI),
Earth-Centered Rotating (ECR/ECEF), and the UNW / TNW local orbital reference
frames, designed for space-based sensor coverage analysis.

All frames route through ECI (J2000) as the canonical inertial base::

    ECR (ECEF)  ←→  ECI (J2000)  ←→  UNW (radial-based)
                                  ←→  TNW (velocity-based)

Coordinate Frame Definitions
-----------------------------

**ECI (Earth-Centered Inertial, J2000)**
  - X: Vernal equinox at J2000.0
  - Z: Mean celestial pole at J2000.0
  - Y: Completes RHS.  Inertial — Newton's laws hold directly.

**ECR (Earth-Centered Rotating / ECEF)**
  - X: Greenwich meridian
  - Z: Geographic pole
  - Y: Completes RHS.  Rotates with Earth at ω_⊕.
  - Full-state transforms include the transport theorem (Coriolis).

**UNW (Radial–Cross-Track–Along-Track, position-based)**
  - U: Unit vector along position (radial outward)
  - W: Along orbital angular momentum (cross-track / orbit-normal)
  - N: Completes RHS (≈ along-track for near-circular orbits)

**TNW (Along-Track–Normal–Cross-Track, velocity-based)**
  - T: Unit vector along velocity (along-track / tangential)
  - N: In-plane, perpendicular to T, toward curvature center
  - W: Along orbital angular momentum (cross-track / orbit-normal)

Author: Generated for Richard's autonomous systems research
"""

from .frames import (
    # ── ECI ↔ ECR ──
    eci_to_ecr_matrix, ecr_to_eci_matrix,
    eci_to_ecr, ecr_to_eci,
    state_eci_to_ecr, state_ecr_to_eci,
    transform_covariance_eci_to_ecr, transform_covariance_ecr_to_eci,
    # ── ECI ↔ UNW ──
    eci_to_unw_matrix, unw_to_eci_matrix,
    eci_to_unw, unw_to_eci,
    state_eci_to_unw,
    transform_covariance_eci_to_unw, transform_covariance_unw_to_eci,
    # ── ECI ↔ TNW ──
    eci_to_tnw_matrix, tnw_to_eci_matrix,
    eci_to_tnw, tnw_to_eci,
    state_eci_to_tnw,
    transform_covariance_eci_to_tnw, transform_covariance_tnw_to_eci,
    # ── Cross-frame (via ECI hub) ──
    unw_to_tnw, tnw_to_unw,
    ecr_to_unw_matrix, unw_to_ecr_matrix,
    ecr_to_unw, unw_to_ecr,
    ecr_to_tnw_matrix, tnw_to_ecr_matrix,
    ecr_to_tnw, tnw_to_ecr,
    # ── Cross-frame covariance ──
    transform_covariance_ecr_to_unw, transform_covariance_unw_to_ecr,
    transform_covariance_ecr_to_tnw, transform_covariance_tnw_to_ecr,
    transform_covariance_unw_to_tnw, transform_covariance_tnw_to_unw,
    # ── Unified API ──
    get_dcm, transform, transform_covariance,
    FRAMES,
)

from .orbits import (
    keplerian_to_eci,
    eci_to_keplerian,
    propagate_kepler,
    compute_orbital_period,
    compute_mean_motion,
)

from .coverage import (
    sensor_footprint_unw,
    sensor_footprint_tnw,
    ground_trace,
    coverage_windows,
    access_geometry,
    earth_intersection,
)

from .utils import (
    normalize,
    rotation_matrix_axis_angle,
    eci_to_ecef, ecef_to_eci,       # legacy aliases (identical to eci_to_ecr / ecr_to_eci)
    ecef_to_lla,
    lla_to_ecef,
    julian_date,
    gmst,
    MU_EARTH,
    R_EARTH,
    J2,
    OMEGA_EARTH,
    F_EARTH,
    E2_EARTH,
)

__version__ = "2.0.0"
__all__ = [
    # ── Constants ──
    "MU_EARTH", "R_EARTH", "J2", "OMEGA_EARTH", "F_EARTH", "E2_EARTH",
    # ── Unified API (recommended entry points) ──
    "get_dcm", "transform", "transform_covariance", "FRAMES",
    # ── ECI ↔ ECR rotation matrices ──
    "eci_to_ecr_matrix", "ecr_to_eci_matrix",
    # ── ECI ↔ ECR vector transforms ──
    "eci_to_ecr", "ecr_to_eci",
    # ── ECI ↔ ECR full-state (with transport theorem) ──
    "state_eci_to_ecr", "state_ecr_to_eci",
    # ── ECI ↔ ECR covariance ──
    "transform_covariance_eci_to_ecr", "transform_covariance_ecr_to_eci",
    # ── ECI ↔ UNW ──
    "eci_to_unw_matrix", "unw_to_eci_matrix",
    "eci_to_unw", "unw_to_eci",
    "state_eci_to_unw",
    "transform_covariance_eci_to_unw", "transform_covariance_unw_to_eci",
    # ── ECI ↔ TNW ──
    "eci_to_tnw_matrix", "tnw_to_eci_matrix",
    "eci_to_tnw", "tnw_to_eci",
    "state_eci_to_tnw",
    "transform_covariance_eci_to_tnw", "transform_covariance_tnw_to_eci",
    # ── Cross-frame ──
    "unw_to_tnw", "tnw_to_unw",
    "ecr_to_unw_matrix", "unw_to_ecr_matrix",
    "ecr_to_unw", "unw_to_ecr",
    "ecr_to_tnw_matrix", "tnw_to_ecr_matrix",
    "ecr_to_tnw", "tnw_to_ecr",
    # ── Cross-frame covariance ──
    "transform_covariance_ecr_to_unw", "transform_covariance_unw_to_ecr",
    "transform_covariance_ecr_to_tnw", "transform_covariance_tnw_to_ecr",
    "transform_covariance_unw_to_tnw", "transform_covariance_tnw_to_unw",
    # ── Orbital mechanics ──
    "keplerian_to_eci", "eci_to_keplerian", "propagate_kepler",
    "compute_orbital_period", "compute_mean_motion",
    # ── Coverage analysis ──
    "sensor_footprint_unw", "sensor_footprint_tnw",
    "ground_trace", "coverage_windows", "access_geometry",
    "earth_intersection",
    # ── Utilities / legacy ──
    "normalize", "rotation_matrix_axis_angle",
    "eci_to_ecef", "ecef_to_eci", "ecef_to_lla", "lla_to_ecef",
    "julian_date", "gmst",
]
