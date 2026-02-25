"""
orbital_frames.frames — Unified ECI / ECR / UNW / TNW Frame Hub
=================================================================

All four coordinate frames route through ECI (J2000) as the canonical base.

Frame Definitions
-----------------

**ECI (Earth-Centered Inertial, J2000)**
  - X: Vernal equinox direction at J2000.0
  - Z: Mean celestial pole at J2000.0
  - Y: Completes right-hand system
  - Inertial frame — Newton's laws hold directly.

**ECR (Earth-Centered Rotating / ECEF)**
  - X: Greenwich meridian (IERS Reference Meridian)
  - Z: Celestial Intermediate Pole (≈ mean geographic pole)
  - Y: Completes right-hand system
  - Rotates with the Earth at ω_⊕ ≈ 7.2921150 × 10⁻⁵ rad/s.
  - **Position-only:** simple GMST rotation.
  - **Full-state (pos + vel):** includes the transport theorem
    (Coriolis term  ω × r) so that velocities are correct in the
    rotating frame.

**UNW (Radial / Cross-Track / Along-Track — position-based)**::

    U = r̂             (radial outward)
    W = (r × v)̂       (orbit-normal, cross-track)
    N = W × U          (≈ along-track for near-circular orbits)

**TNW (Along-Track / Normal / Cross-Track — velocity-based)**::

    T = v̂             (along-track / tangential)
    W = (r × v)̂       (orbit-normal, cross-track)
    N = W × T          (in-plane normal, toward curvature center)

Transform Graph
---------------
All transforms route through ECI::

    ECR ←→ ECI ←→ UNW
                ←→ TNW

Any-to-any transforms compose through ECI automatically:
  ECR→TNW = (ECI→TNW) ∘ (ECR→ECI)
  UNW→ECR = (ECI→ECR) ∘ (UNW→ECI)
"""

import numpy as np
from numpy.typing import NDArray

from .utils import normalize, gmst, OMEGA_EARTH


# ════════════════════════════════════════════════════════════════════════════
#  Internal Helpers
# ════════════════════════════════════════════════════════════════════════════

def _apply_dcm(R: NDArray, vec: NDArray) -> NDArray:
    """Apply 3×3 DCM to a single (3,) or batch (N,3) of vectors."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim == 1:
        return R @ vec
    return (R @ vec.T).T


def _build_6x6(R: NDArray) -> NDArray:
    """Expand a 3×3 DCM to block-diagonal 6×6 for full-state transforms."""
    M = np.zeros((6, 6), dtype=np.float64)
    M[:3, :3] = R
    M[3:, 3:] = R
    return M


def _transform_covariance(P: NDArray, R: NDArray) -> NDArray:
    """Generic covariance rotation: P' = R P Rᵀ.

    Handles 3×3 (position-only) and 6×6 (full-state) covariance matrices.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape == (3, 3):
        return R @ P @ R.T
    elif P.shape == (6, 6):
        M = _build_6x6(R)
        return M @ P @ M.T
    else:
        raise ValueError(f"Covariance must be (3,3) or (6,6), got {P.shape}")


# ════════════════════════════════════════════════════════════════════════════
#  ECI ↔ ECR  (Earth-Centered Rotating / ECEF)
# ════════════════════════════════════════════════════════════════════════════
#
#  ECR differs from the orbital-local frames because it requires a time
#  parameter (Julian Date) rather than an orbital state, and the velocity
#  transform includes the transport theorem Coriolis correction.
#
#  Position :  r_ecr = R_z(θ) · r_eci
#  Velocity :  v_ecr = R_z(θ) · (v_eci − ω×r_eci)
#
#  where θ = GMST(jd) and ω = [0, 0, ω_⊕]
# ════════════════════════════════════════════════════════════════════════════

def eci_to_ecr_matrix(jd: float) -> NDArray:
    """Build the ECI→ECR (ECEF) 3×3 rotation matrix at a given Julian Date.

    This is a simple z-rotation by GMST.  For position-only transforms.

    Parameters
    ----------
    jd : float — Julian Date (UTC)

    Returns
    -------
    R : (3,3) ndarray — DCM such that r_ecr = R @ r_eci
    """
    theta = gmst(jd)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def ecr_to_eci_matrix(jd: float) -> NDArray:
    """ECR→ECI 3×3 rotation matrix (transpose of ECI→ECR)."""
    return eci_to_ecr_matrix(jd).T


def eci_to_ecr(vec_eci: NDArray, jd: float) -> NDArray:
    """Transform position vector(s) from ECI to ECR.

    Parameters
    ----------
    vec_eci : (3,) or (N,3) — position(s) in ECI [m]
    jd : float — Julian Date

    Returns
    -------
    vec_ecr : same shape — position(s) in ECR [m]
    """
    return _apply_dcm(eci_to_ecr_matrix(jd), vec_eci)


def ecr_to_eci(vec_ecr: NDArray, jd: float) -> NDArray:
    """Transform position vector(s) from ECR to ECI."""
    return _apply_dcm(ecr_to_eci_matrix(jd), vec_ecr)


def state_eci_to_ecr(r_eci: NDArray, v_eci: NDArray,
                     jd: float) -> tuple[NDArray, NDArray]:
    """Transform full state (position + velocity) from ECI to ECR.

    Applies the transport theorem::

        v_ecr = R · (v_eci − ω × r_eci)

    Parameters
    ----------
    r_eci : (3,) — position in ECI [m]
    v_eci : (3,) — velocity in ECI [m/s]
    jd : float — Julian Date

    Returns
    -------
    r_ecr : (3,) — position in ECR [m]
    v_ecr : (3,) — velocity in ECR [m/s]
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)
    R = eci_to_ecr_matrix(jd)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])

    r_ecr = R @ r
    v_ecr = R @ (v - np.cross(omega, r))
    return r_ecr, v_ecr


def state_ecr_to_eci(r_ecr: NDArray, v_ecr: NDArray,
                     jd: float) -> tuple[NDArray, NDArray]:
    """Transform full state from ECR to ECI (inverse transport theorem).

    ::

        v_eci = Rᵀ · v_ecr + ω × r_eci

    Parameters
    ----------
    r_ecr : (3,) — position in ECR [m]
    v_ecr : (3,) — velocity in ECR [m/s]
    jd : float — Julian Date

    Returns
    -------
    r_eci : (3,) — position in ECI [m]
    v_eci : (3,) — velocity in ECI [m/s]
    """
    r_e = np.asarray(r_ecr, dtype=np.float64)
    v_e = np.asarray(v_ecr, dtype=np.float64)
    R_inv = ecr_to_eci_matrix(jd)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])

    r_eci = R_inv @ r_e
    v_eci = R_inv @ v_e + np.cross(omega, r_eci)
    return r_eci, v_eci


def transform_covariance_eci_to_ecr(
    P_eci: NDArray, jd: float
) -> NDArray:
    """Transform covariance from ECI to ECR.

    Note: This is the position-rotation-only covariance transform
    (ignores Coriolis coupling in the velocity partition).  Sufficient
    for position-only covariances and a good approximation for short arcs.
    """
    R = eci_to_ecr_matrix(jd)
    return _transform_covariance(P_eci, R)


def transform_covariance_ecr_to_eci(
    P_ecr: NDArray, jd: float
) -> NDArray:
    """Transform covariance from ECR to ECI."""
    R = ecr_to_eci_matrix(jd)
    return _transform_covariance(P_ecr, R)


# ════════════════════════════════════════════════════════════════════════════
#  ECI ↔ UNW
# ════════════════════════════════════════════════════════════════════════════

def eci_to_unw_matrix(r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Build the ECI→UNW direction-cosine matrix.

    Parameters
    ----------
    r_eci : (3,) array — position in ECI [m]
    v_eci : (3,) array — velocity in ECI [m/s]

    Returns
    -------
    R : (3,3) ndarray — DCM such that v_unw = R @ v_eci
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)

    U = normalize(r)                    # radial outward
    h = np.cross(r, v)                  # angular momentum vector
    W = normalize(h)                    # orbit-normal (cross-track)
    N = np.cross(W, U)                  # ~ along-track (completes RHS)

    return np.array([U, N, W])          # rows = local-frame axes in ECI


def unw_to_eci_matrix(r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """UNW→ECI DCM (transpose of ECI→UNW)."""
    return eci_to_unw_matrix(r_eci, v_eci).T


def eci_to_unw(vec_eci: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform vector(s) from ECI to UNW."""
    return _apply_dcm(eci_to_unw_matrix(r_eci, v_eci), vec_eci)


def unw_to_eci(vec_unw: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform vector(s) from UNW to ECI."""
    return _apply_dcm(unw_to_eci_matrix(r_eci, v_eci), vec_unw)


def state_eci_to_unw(state_eci: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform a 6-element [r; v] state from ECI to UNW."""
    R = eci_to_unw_matrix(r_eci, v_eci)
    return _build_6x6(R) @ np.asarray(state_eci, dtype=np.float64)


def transform_covariance_eci_to_unw(
    P_eci: NDArray, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance matrix from ECI to UNW frame."""
    R = eci_to_unw_matrix(r_eci, v_eci)
    return _transform_covariance(P_eci, R)


def transform_covariance_unw_to_eci(
    P_unw: NDArray, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance matrix from UNW to ECI frame."""
    R = unw_to_eci_matrix(r_eci, v_eci)
    return _transform_covariance(P_unw, R)


# ════════════════════════════════════════════════════════════════════════════
#  ECI ↔ TNW
# ════════════════════════════════════════════════════════════════════════════

def eci_to_tnw_matrix(r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Build the ECI→TNW direction-cosine matrix.

    Parameters
    ----------
    r_eci : (3,) array — position in ECI [m]
    v_eci : (3,) array — velocity in ECI [m/s]

    Returns
    -------
    R : (3,3) ndarray — DCM such that v_tnw = R @ v_eci
    """
    r = np.asarray(r_eci, dtype=np.float64)
    v = np.asarray(v_eci, dtype=np.float64)

    T = normalize(v)                    # along-track (tangential)
    h = np.cross(r, v)
    W = normalize(h)                    # orbit-normal (cross-track)
    N = np.cross(W, T)                  # in-plane normal

    return np.array([T, N, W])


def tnw_to_eci_matrix(r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """TNW→ECI DCM (transpose of ECI→TNW)."""
    return eci_to_tnw_matrix(r_eci, v_eci).T


def eci_to_tnw(vec_eci: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform vector(s) from ECI to TNW."""
    return _apply_dcm(eci_to_tnw_matrix(r_eci, v_eci), vec_eci)


def tnw_to_eci(vec_tnw: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform vector(s) from TNW to ECI."""
    return _apply_dcm(tnw_to_eci_matrix(r_eci, v_eci), vec_tnw)


def state_eci_to_tnw(state_eci: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform a 6-element [r; v] state from ECI to TNW."""
    R = eci_to_tnw_matrix(r_eci, v_eci)
    return _build_6x6(R) @ np.asarray(state_eci, dtype=np.float64)


def transform_covariance_eci_to_tnw(
    P_eci: NDArray, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance matrix from ECI to TNW frame."""
    R = eci_to_tnw_matrix(r_eci, v_eci)
    return _transform_covariance(P_eci, R)


def transform_covariance_tnw_to_eci(
    P_tnw: NDArray, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance matrix from TNW to ECI frame."""
    R = tnw_to_eci_matrix(r_eci, v_eci)
    return _transform_covariance(P_tnw, R)


# ════════════════════════════════════════════════════════════════════════════
#  Cross-Frame Transforms (via ECI hub)
# ════════════════════════════════════════════════════════════════════════════

# ── UNW ↔ TNW ──

def unw_to_tnw(vec_unw: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform vector(s) from UNW to TNW (via ECI)."""
    R = eci_to_tnw_matrix(r_eci, v_eci) @ unw_to_eci_matrix(r_eci, v_eci)
    return _apply_dcm(R, vec_unw)


def tnw_to_unw(vec_tnw: NDArray, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform vector(s) from TNW to UNW (via ECI)."""
    R = eci_to_unw_matrix(r_eci, v_eci) @ tnw_to_eci_matrix(r_eci, v_eci)
    return _apply_dcm(R, vec_tnw)


# ── ECR ↔ UNW ──

def ecr_to_unw_matrix(jd: float, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """ECR→UNW composite DCM: R_eci2unw · R_ecr2eci."""
    return eci_to_unw_matrix(r_eci, v_eci) @ ecr_to_eci_matrix(jd)


def unw_to_ecr_matrix(jd: float, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """UNW→ECR composite DCM: R_eci2ecr · R_unw2eci."""
    return eci_to_ecr_matrix(jd) @ unw_to_eci_matrix(r_eci, v_eci)


def ecr_to_unw(vec_ecr: NDArray, jd: float,
               r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform position vector(s) from ECR to UNW.

    Parameters
    ----------
    vec_ecr : (3,) or (N,3) — position(s) in ECR [m]
    jd : float — Julian Date
    r_eci, v_eci : (3,) — reference satellite state in ECI
    """
    return _apply_dcm(ecr_to_unw_matrix(jd, r_eci, v_eci), vec_ecr)


def unw_to_ecr(vec_unw: NDArray, jd: float,
               r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform position vector(s) from UNW to ECR."""
    return _apply_dcm(unw_to_ecr_matrix(jd, r_eci, v_eci), vec_unw)


# ── ECR ↔ TNW ──

def ecr_to_tnw_matrix(jd: float, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """ECR→TNW composite DCM: R_eci2tnw · R_ecr2eci."""
    return eci_to_tnw_matrix(r_eci, v_eci) @ ecr_to_eci_matrix(jd)


def tnw_to_ecr_matrix(jd: float, r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """TNW→ECR composite DCM: R_eci2ecr · R_tnw2eci."""
    return eci_to_ecr_matrix(jd) @ tnw_to_eci_matrix(r_eci, v_eci)


def ecr_to_tnw(vec_ecr: NDArray, jd: float,
               r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform position vector(s) from ECR to TNW."""
    return _apply_dcm(ecr_to_tnw_matrix(jd, r_eci, v_eci), vec_ecr)


def tnw_to_ecr(vec_tnw: NDArray, jd: float,
               r_eci: NDArray, v_eci: NDArray) -> NDArray:
    """Transform position vector(s) from TNW to ECR."""
    return _apply_dcm(tnw_to_ecr_matrix(jd, r_eci, v_eci), vec_tnw)


# ── Cross-Frame Covariance ──

def transform_covariance_ecr_to_unw(
    P_ecr: NDArray, jd: float, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance from ECR to UNW (via composed DCM)."""
    R = ecr_to_unw_matrix(jd, r_eci, v_eci)
    return _transform_covariance(P_ecr, R)


def transform_covariance_unw_to_ecr(
    P_unw: NDArray, jd: float, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance from UNW to ECR."""
    R = unw_to_ecr_matrix(jd, r_eci, v_eci)
    return _transform_covariance(P_unw, R)


def transform_covariance_ecr_to_tnw(
    P_ecr: NDArray, jd: float, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance from ECR to TNW."""
    R = ecr_to_tnw_matrix(jd, r_eci, v_eci)
    return _transform_covariance(P_ecr, R)


def transform_covariance_tnw_to_ecr(
    P_tnw: NDArray, jd: float, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance from TNW to ECR."""
    R = tnw_to_ecr_matrix(jd, r_eci, v_eci)
    return _transform_covariance(P_tnw, R)


def transform_covariance_unw_to_tnw(
    P_unw: NDArray, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance from UNW to TNW."""
    R = eci_to_tnw_matrix(r_eci, v_eci) @ unw_to_eci_matrix(r_eci, v_eci)
    return _transform_covariance(P_unw, R)


def transform_covariance_tnw_to_unw(
    P_tnw: NDArray, r_eci: NDArray, v_eci: NDArray
) -> NDArray:
    """Transform covariance from TNW to UNW."""
    R = eci_to_unw_matrix(r_eci, v_eci) @ tnw_to_eci_matrix(r_eci, v_eci)
    return _transform_covariance(P_tnw, R)


# ════════════════════════════════════════════════════════════════════════════
#  Unified Transform API
# ════════════════════════════════════════════════════════════════════════════

# Valid frame names
FRAMES = {"eci", "ecr", "unw", "tnw"}


def get_dcm(from_frame: str, to_frame: str,
            jd: float | None = None,
            r_eci: NDArray | None = None,
            v_eci: NDArray | None = None) -> NDArray:
    """Get the 3×3 DCM for any supported frame pair.

    This is the universal entry point: specify source and target frame
    by name, provide the required context, and get back the rotation matrix.

    Parameters
    ----------
    from_frame : str — one of 'eci', 'ecr', 'unw', 'tnw'
    to_frame : str — one of 'eci', 'ecr', 'unw', 'tnw'
    jd : float or None — Julian Date (required if ECR is involved)
    r_eci, v_eci : (3,) or None — satellite ECI state (required if UNW/TNW)

    Returns
    -------
    R : (3,3) ndarray — DCM such that v_to = R @ v_from
    """
    fr = from_frame.lower()
    to = to_frame.lower()
    if fr not in FRAMES or to not in FRAMES:
        raise ValueError(f"Unknown frame. Valid: {FRAMES}")
    if fr == to:
        return np.eye(3)

    # Step 1: from_frame → ECI
    if fr == "eci":
        R_to_eci = np.eye(3)
    elif fr == "ecr":
        if jd is None:
            raise ValueError("jd required for ECR transforms")
        R_to_eci = ecr_to_eci_matrix(jd)
    elif fr == "unw":
        if r_eci is None or v_eci is None:
            raise ValueError("r_eci, v_eci required for UNW transforms")
        R_to_eci = unw_to_eci_matrix(r_eci, v_eci)
    elif fr == "tnw":
        if r_eci is None or v_eci is None:
            raise ValueError("r_eci, v_eci required for TNW transforms")
        R_to_eci = tnw_to_eci_matrix(r_eci, v_eci)

    # Step 2: ECI → to_frame
    if to == "eci":
        R_from_eci = np.eye(3)
    elif to == "ecr":
        if jd is None:
            raise ValueError("jd required for ECR transforms")
        R_from_eci = eci_to_ecr_matrix(jd)
    elif to == "unw":
        if r_eci is None or v_eci is None:
            raise ValueError("r_eci, v_eci required for UNW transforms")
        R_from_eci = eci_to_unw_matrix(r_eci, v_eci)
    elif to == "tnw":
        if r_eci is None or v_eci is None:
            raise ValueError("r_eci, v_eci required for TNW transforms")
        R_from_eci = eci_to_tnw_matrix(r_eci, v_eci)

    return R_from_eci @ R_to_eci


def transform(vec: NDArray,
              from_frame: str, to_frame: str,
              jd: float | None = None,
              r_eci: NDArray | None = None,
              v_eci: NDArray | None = None) -> NDArray:
    """Transform position vector(s) between any two frames.

    Parameters
    ----------
    vec : (3,) or (N,3) — position vector(s) in from_frame
    from_frame, to_frame : str — frame names ('eci','ecr','unw','tnw')
    jd : float or None — Julian Date (needed if ECR involved)
    r_eci, v_eci : (3,) or None — satellite ECI state (needed if UNW/TNW)

    Returns
    -------
    vec_out : same shape — vector(s) in to_frame
    """
    R = get_dcm(from_frame, to_frame, jd=jd, r_eci=r_eci, v_eci=v_eci)
    return _apply_dcm(R, vec)


def transform_covariance(P: NDArray,
                         from_frame: str, to_frame: str,
                         jd: float | None = None,
                         r_eci: NDArray | None = None,
                         v_eci: NDArray | None = None) -> NDArray:
    """Transform covariance matrix between any two frames.

    Parameters
    ----------
    P : (3,3) or (6,6) — covariance in from_frame
    from_frame, to_frame : str — frame names
    jd, r_eci, v_eci : context (same rules as get_dcm)

    Returns
    -------
    P_out : same shape — covariance in to_frame
    """
    R = get_dcm(from_frame, to_frame, jd=jd, r_eci=r_eci, v_eci=v_eci)
    return _transform_covariance(P, R)
