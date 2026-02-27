#!/usr/bin/env python3
"""
test_astrosor.py — Comprehensive Unit Test Suite
=======================================================

Tests all modules of the astrosor library:

  1. utils       — constants, vector ops, time, coordinate conversions
  2. frames      — ECI/ECR/UNW/TNW rotation matrices, transforms, covariance
  3. orbits      — Keplerian conversion, propagation, energy/momentum
  4. coverage    — footprints, ground trace, access geometry, coverage windows
  5. sun         — ephemeris, exclusion, eclipse, phase angle
  6. moon        — ephemeris, exclusion, illumination, phase
  7. exclusion   — combined exclusion, availability timeline, observability
  8. tle         — TLE parsing, epoch state, propagation
  9. sensor      — ground sensor model, azel, visibility, visual magnitude
 10. scheduler   — priority scoring, greedy scheduling, format output

Run:  python3 test_astrosor.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import numpy.testing as npt

from astrosor import *
from astrosor.frames import (
    state_eci_to_ecr, state_ecr_to_eci,
    ecr_to_unw_matrix, unw_to_ecr_matrix,
    ecr_to_tnw_matrix, tnw_to_ecr_matrix,
    _build_6x6,
)
from astrosor.orbits import propagate_j2
from astrosor.sun import eclipse_conical, eclipse_cylindrical, sep_angle
from astrosor.moon import (
    moon_phase_angle, moon_age_days, moon_phase_name,
    moon_angular_radius, sun_moon_angle, moon_earth_shadow,
)
from astrosor.exclusion import check_all_exclusions, target_observability
from astrosor.sensor import (
    GroundSensor, check_visibility, topocentric_azel,
    sun_elevation_at_sensor, estimate_visual_magnitude,
)
from astrosor.tle import TLE, parse_tle, parse_tle_batch, tle_epoch_state, propagate_tle
from astrosor.scheduler import (
    SatelliteTask, ScheduleResult,
    compute_visibility_windows, compute_urgency, compute_score,
    schedule_greedy, format_schedule,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════

JD_TEST = julian_date(2026, 2, 24, 12, 0, 0)

# Circular LEO (ISS-like)
_LEO = keplerian_to_eci(R_EARTH + 400e3, 0.0001, np.deg2rad(51.6),
                        np.deg2rad(30), 0.0, np.deg2rad(45))
R_LEO, V_LEO = _LEO

# Elliptical HEO (Molniya-like)
_HEO = keplerian_to_eci(26_600e3, 0.74, np.deg2rad(63.4), 0.0,
                        np.deg2rad(270), np.deg2rad(90))
R_HEO, V_HEO = _HEO

# Reference ISS TLE
ISS_NAME = "ISS (ZARYA)"
ISS_L1 = "1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  9002"
ISS_L2 = "2 25544  51.6400 210.0000 0007000  90.0000 270.0000 15.50000000400000"

# Reference sensors
SENSOR_RADAR = GroundSensor(
    name="TestRadar", lat=np.deg2rad(42.62), lon=np.deg2rad(-71.49),
    alt=146.0, sensor_type="radar", min_elevation=np.deg2rad(5.0),
    max_range=40_000e3,
)
SENSOR_OPTICAL = GroundSensor(
    name="TestOptical", lat=np.deg2rad(32.0), lon=np.deg2rad(-106.0),
    alt=1300.0, sensor_type="optical", limiting_magnitude=18.0,
    min_elevation=np.deg2rad(10.0),
    solar_exclusion_angle=np.deg2rad(30.0),
    lunar_exclusion_angle=np.deg2rad(10.0),
)


# ═══════════════════════════════════════════════════════════════════════════
#  Test Runner Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

_results = {"pass": 0, "fail": 0, "errors": []}


def run_test(name, fn):
    """Execute a single test, track results."""
    try:
        fn()
        print(f"  PASS  {name}")
        _results["pass"] += 1
    except Exception as ex:
        print(f"  FAIL  {name}")
        print(f"         {ex}")
        traceback.print_exc(limit=3)
        _results["fail"] += 1
        _results["errors"].append(name)


# ═══════════════════════════════════════════════════════════════════════════
#  1. UTILS MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_utils():
    print("\n── utils ──")

    def t_normalize_unit():
        v = np.array([3.0, 4.0, 0.0])
        npt.assert_allclose(np.linalg.norm(normalize(v)), 1.0, atol=1e-15)
    run_test("normalize → unit length", t_normalize_unit)

    def t_normalize_direction():
        v = np.array([3.0, 4.0, 0.0])
        npt.assert_allclose(normalize(v), [0.6, 0.8, 0.0], atol=1e-15)
    run_test("normalize direction correct", t_normalize_direction)

    def t_normalize_batch():
        vecs = np.random.randn(20, 3) * 100
        normed = normalize(vecs)
        npt.assert_allclose(np.linalg.norm(normed, axis=1), np.ones(20), atol=1e-14)
    run_test("normalize batch (N,3)", t_normalize_batch)

    def t_normalize_zero_raises():
        try:
            normalize(np.zeros(3))
            assert False, "Should raise"
        except ValueError:
            pass
    run_test("normalize zero → ValueError", t_normalize_zero_raises)

    def t_rotation_identity():
        R = rotation_matrix_axis_angle(np.array([0, 0, 1]), 0.0)
        npt.assert_allclose(R, np.eye(3), atol=1e-14)
    run_test("rotation 0° = identity", t_rotation_identity)

    def t_rotation_90z():
        R = rotation_matrix_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        npt.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-14)
    run_test("rotation 90° about Z: x̂→ŷ", t_rotation_90z)

    def t_rotation_180():
        R = rotation_matrix_axis_angle(np.array([0, 0, 1]), np.pi)
        npt.assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-14)
    run_test("rotation 180° about Z: x̂→−x̂", t_rotation_180)

    def t_rotation_orthogonal():
        R = rotation_matrix_axis_angle(np.array([1, 1, 1]), 1.23)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    run_test("rotation matrix orthogonal + det=1", t_rotation_orthogonal)

    def t_jd_j2000():
        npt.assert_allclose(julian_date(2000, 1, 1, 12), 2451545.0, atol=1e-6)
    run_test("JD at J2000.0 = 2451545.0", t_jd_j2000)

    def t_jd_known():
        jd = julian_date(2026, 2, 24, 0, 0, 0)
        assert abs(jd - 2461095.5) < 1.0
    run_test("JD known date sanity", t_jd_known)

    def t_jd_monotonic():
        jd1 = julian_date(2026, 1, 1, 0)
        jd2 = julian_date(2026, 6, 1, 0)
        assert jd2 > jd1
    run_test("JD monotonically increasing", t_jd_monotonic)

    def t_gmst_range():
        g = gmst(JD_TEST)
        assert 0 <= g < 2 * np.pi
    run_test("GMST in [0, 2π)", t_gmst_range)

    def t_gmst_increases():
        g1 = gmst(JD_TEST)
        g2 = gmst(JD_TEST + 0.01)  # ~15 min later
        # GMST should change (Earth rotates)
        assert abs(g2 - g1) > 0
    run_test("GMST changes with time", t_gmst_increases)

    def t_eci_ecef_roundtrip():
        r_ecef = eci_to_ecef(R_LEO, JD_TEST)
        r_back = ecef_to_eci(r_ecef, JD_TEST)
        npt.assert_allclose(r_back, R_LEO, atol=1e-6)
    run_test("ECI↔ECEF roundtrip", t_eci_ecef_roundtrip)

    def t_eci_ecef_magnitude():
        r_ecef = eci_to_ecef(R_LEO, JD_TEST)
        npt.assert_allclose(np.linalg.norm(r_ecef), np.linalg.norm(R_LEO), atol=1e-8)
    run_test("ECEF preserves magnitude", t_eci_ecef_magnitude)

    def t_eci_ecef_batch():
        vecs = np.random.randn(30, 3) * R_EARTH
        ecef = eci_to_ecef(vecs, JD_TEST)
        back = ecef_to_eci(ecef, JD_TEST)
        npt.assert_allclose(back, vecs, atol=1e-6)
    run_test("ECI↔ECEF batch (30,3)", t_eci_ecef_batch)

    def t_lla_equator():
        r = lla_to_ecef(0.0, 0.0, 0.0)
        npt.assert_allclose(r[0], R_EARTH, rtol=1e-10)
        npt.assert_allclose(r[1], 0.0, atol=1e-6)
        npt.assert_allclose(r[2], 0.0, atol=1e-6)
    run_test("LLA equator → ECEF x-axis", t_lla_equator)

    def t_lla_pole():
        r = lla_to_ecef(np.pi / 2, 0.0, 0.0)
        npt.assert_allclose(r[0], 0.0, atol=1e-3)
        npt.assert_allclose(r[1], 0.0, atol=1e-3)
        assert r[2] > 0
    run_test("LLA north pole → +Z", t_lla_pole)

    def t_lla_roundtrip():
        lat, lon, alt = np.deg2rad(40), np.deg2rad(-74), 100.0
        r = lla_to_ecef(lat, lon, alt)
        lla = ecef_to_lla(r)
        npt.assert_allclose(lla[0], lat, atol=1e-10)
        npt.assert_allclose(lla[1], lon, atol=1e-10)
        npt.assert_allclose(lla[2], alt, atol=0.01)
    run_test("LLA↔ECEF roundtrip", t_lla_roundtrip)

    def t_lla_roundtrip_batch():
        lats = np.deg2rad(np.array([-80, -30, 0, 30, 80]))
        lons = np.deg2rad(np.array([0, 45, 90, 135, 180]))
        pts = np.array([lla_to_ecef(la, lo, 5000.0) for la, lo in zip(lats, lons)])
        lla = ecef_to_lla(pts)
        npt.assert_allclose(lla[:, 0], lats, atol=1e-9)
        npt.assert_allclose(lla[:, 1], lons, atol=1e-9)
    run_test("LLA↔ECEF batch roundtrip", t_lla_roundtrip_batch)

    def t_lla_altitude():
        alt = 50000.0  # 50 km
        r = lla_to_ecef(np.deg2rad(45), np.deg2rad(90), alt)
        lla = ecef_to_lla(r)
        npt.assert_allclose(lla[2], alt, atol=0.1)
    run_test("LLA altitude preserved", t_lla_altitude)

    def t_constants():
        assert MU_EARTH > 3.9e14
        assert R_EARTH > 6.37e6
        assert 0 < J2 < 0.01
        assert OMEGA_EARTH > 7e-5
        assert 0 < F_EARTH < 0.01
        assert 0 < E2_EARTH < 0.01
    run_test("physical constants sanity", t_constants)


# ═══════════════════════════════════════════════════════════════════════════
#  2. FRAMES MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_frames():
    print("\n── frames ──")

    # ── UNW ──
    def t_unw_orthogonal():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    run_test("UNW DCM orthogonal + det=1", t_unw_orthogonal)

    def t_unw_u_radial():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[0], normalize(R_LEO), atol=1e-14)
    run_test("UNW U-axis = radial", t_unw_u_radial)

    def t_unw_w_orbit_normal():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        h = np.cross(R_LEO, V_LEO)
        npt.assert_allclose(R[2], normalize(h), atol=1e-14)
    run_test("UNW W-axis = orbit normal", t_unw_w_orbit_normal)

    def t_unw_n_completes_rhs():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[1], np.cross(R[2], R[0]), atol=1e-14)
    run_test("UNW N = W×U (right-hand)", t_unw_n_completes_rhs)

    def t_unw_elliptical():
        R = eci_to_unw_matrix(R_HEO, V_HEO)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    run_test("UNW orthogonal (elliptical orbit)", t_unw_elliptical)

    # ── TNW ──
    def t_tnw_orthogonal():
        R = eci_to_tnw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    run_test("TNW DCM orthogonal + det=1", t_tnw_orthogonal)

    def t_tnw_t_velocity():
        R = eci_to_tnw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[0], normalize(V_LEO), atol=1e-14)
    run_test("TNW T-axis = velocity", t_tnw_t_velocity)

    def t_tnw_n_completes():
        R = eci_to_tnw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[1], np.cross(R[2], R[0]), atol=1e-14)
    run_test("TNW N = W×T (right-hand)", t_tnw_n_completes)

    def t_w_axes_match():
        npt.assert_allclose(
            eci_to_unw_matrix(R_LEO, V_LEO)[2],
            eci_to_tnw_matrix(R_LEO, V_LEO)[2], atol=1e-14)
    run_test("UNW.W == TNW.W (shared orbit normal)", t_w_axes_match)

    def t_unw_tnw_diverge_eccentric():
        # For eccentric orbits at ν≠0,180, UNW and TNW should differ
        R_unw = eci_to_unw_matrix(R_HEO, V_HEO)
        R_tnw = eci_to_tnw_matrix(R_HEO, V_HEO)
        # T and N_unw should NOT be identical
        cos_ang = np.clip(np.dot(R_tnw[0], R_unw[1]), -1, 1)
        angle = np.arccos(cos_ang)
        assert angle > np.deg2rad(5), f"UNW/TNW divergence only {np.rad2deg(angle):.1f}°"
    run_test("UNW/TNW diverge for eccentric orbit", t_unw_tnw_diverge_eccentric)

    # ── ECR ──
    def t_ecr_orthogonal():
        R = eci_to_ecr_matrix(JD_TEST)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    run_test("ECR DCM orthogonal", t_ecr_orthogonal)

    def t_ecr_z_invariant():
        R = eci_to_ecr_matrix(JD_TEST)
        npt.assert_allclose(R[2], [0, 0, 1], atol=1e-14)
    run_test("ECR Z-axis invariant (pole unchanged)", t_ecr_z_invariant)

    def t_ecr_inverse():
        R = eci_to_ecr_matrix(JD_TEST)
        npt.assert_allclose(R @ ecr_to_eci_matrix(JD_TEST), np.eye(3), atol=1e-14)
    run_test("ECR R·R⁻¹ = I", t_ecr_inverse)

    # ── Vector roundtrips ──
    def t_eci_unw_rt():
        v = np.array([1e3, -2e3, 3e3])
        npt.assert_allclose(unw_to_eci(eci_to_unw(v, R_LEO, V_LEO), R_LEO, V_LEO), v, atol=1e-8)
    run_test("ECI→UNW→ECI roundtrip", t_eci_unw_rt)

    def t_eci_tnw_rt():
        v = np.array([1e3, -2e3, 3e3])
        npt.assert_allclose(tnw_to_eci(eci_to_tnw(v, R_LEO, V_LEO), R_LEO, V_LEO), v, atol=1e-8)
    run_test("ECI→TNW→ECI roundtrip", t_eci_tnw_rt)

    def t_unw_tnw_rt():
        v = np.array([100, 200, 300])
        npt.assert_allclose(tnw_to_unw(unw_to_tnw(v, R_LEO, V_LEO), R_LEO, V_LEO), v, atol=1e-8)
    run_test("UNW→TNW→UNW roundtrip", t_unw_tnw_rt)

    def t_eci_ecr_rt():
        npt.assert_allclose(ecr_to_eci(eci_to_ecr(R_LEO, JD_TEST), JD_TEST), R_LEO, atol=1e-8)
    run_test("ECI→ECR→ECI roundtrip", t_eci_ecr_rt)

    def t_ecr_unw_rt():
        v = np.array([1e6, 2e6, 3e6])
        npt.assert_allclose(
            unw_to_ecr(ecr_to_unw(v, JD_TEST, R_LEO, V_LEO), JD_TEST, R_LEO, V_LEO),
            v, atol=1e-6)
    run_test("ECR→UNW→ECR roundtrip", t_ecr_unw_rt)

    def t_ecr_tnw_rt():
        v = np.array([1e6, 2e6, 3e6])
        npt.assert_allclose(
            tnw_to_ecr(ecr_to_tnw(v, JD_TEST, R_LEO, V_LEO), JD_TEST, R_LEO, V_LEO),
            v, atol=1e-6)
    run_test("ECR→TNW→ECR roundtrip", t_ecr_tnw_rt)

    def t_batch_transform():
        vecs = np.random.randn(50, 3) * 1000
        npt.assert_allclose(unw_to_eci(eci_to_unw(vecs, R_LEO, V_LEO), R_LEO, V_LEO), vecs, atol=1e-8)
    run_test("batch (50,3) UNW roundtrip", t_batch_transform)

    def t_magnitude_preserved():
        v = np.array([5e3, -3e3, 7e3])
        npt.assert_allclose(np.linalg.norm(eci_to_unw(v, R_LEO, V_LEO)), np.linalg.norm(v), atol=1e-8)
    run_test("vector magnitude preserved", t_magnitude_preserved)

    # ── State transforms (transport theorem) ──
    def t_state_ecr_rt():
        r_ecr, v_ecr = state_eci_to_ecr(R_LEO, V_LEO, JD_TEST)
        r_back, v_back = state_ecr_to_eci(r_ecr, v_ecr, JD_TEST)
        npt.assert_allclose(r_back, R_LEO, atol=1e-8)
        npt.assert_allclose(v_back, V_LEO, atol=1e-8)
    run_test("state ECI↔ECR roundtrip (transport theorem)", t_state_ecr_rt)

    def t_transport_coriolis():
        v_ecr_wrong = eci_to_ecr(V_LEO, JD_TEST)
        _, v_ecr_right = state_eci_to_ecr(R_LEO, V_LEO, JD_TEST)
        diff = np.linalg.norm(v_ecr_right - v_ecr_wrong)
        assert 1.0 < diff < 600, f"Coriolis correction {diff:.1f} m/s"
    run_test("transport theorem Coriolis non-trivial", t_transport_coriolis)

    def t_ground_station_zero_ecr_vel():
        r_gs_ecef = lla_to_ecef(np.deg2rad(40), np.deg2rad(-74), 0)
        r_gs_eci = ecef_to_eci(r_gs_ecef, JD_TEST)
        omega = np.array([0, 0, OMEGA_EARTH])
        v_gs_eci = np.cross(omega, r_gs_eci)
        _, v_ecr = state_eci_to_ecr(r_gs_eci, v_gs_eci, JD_TEST)
        npt.assert_allclose(v_ecr, np.zeros(3), atol=1e-4)
    run_test("ground station → zero ECR velocity", t_ground_station_zero_ecr_vel)

    def t_state_unw():
        state = np.concatenate([R_LEO, V_LEO]) + np.array([100, 200, 300, 1, 2, 3], dtype=float)
        s_unw = state_eci_to_unw(state, R_LEO, V_LEO)
        assert s_unw.shape == (6,)
    run_test("state_eci_to_unw 6-vector", t_state_unw)

    # ── Covariance ──
    def t_cov3_unw_rt():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_unw_to_eci(transform_covariance_eci_to_unw(P, R_LEO, V_LEO), R_LEO, V_LEO),
            P, atol=1e-6)
    run_test("cov 3×3 ECI↔UNW roundtrip", t_cov3_unw_rt)

    def t_cov6_tnw_rt():
        A = np.random.randn(6, 6) * 100; P = A @ A.T + np.eye(6) * 10
        npt.assert_allclose(
            transform_covariance_tnw_to_eci(transform_covariance_eci_to_tnw(P, R_LEO, V_LEO), R_LEO, V_LEO),
            P, atol=1e-4)
    run_test("cov 6×6 ECI↔TNW roundtrip", t_cov6_tnw_rt)

    def t_cov_ecr_rt():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_ecr_to_eci(transform_covariance_eci_to_ecr(P, JD_TEST), JD_TEST),
            P, atol=1e-6)
    run_test("cov 3×3 ECI↔ECR roundtrip", t_cov_ecr_rt)

    def t_cov_trace():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(np.trace(P), np.trace(transform_covariance_eci_to_unw(P, R_LEO, V_LEO)), rtol=1e-12)
    run_test("cov trace preserved under rotation", t_cov_trace)

    def t_cov_positive_definite():
        A = np.random.randn(3, 3) * 50; P = A @ A.T + np.eye(3) * 5
        eigvals = np.linalg.eigvalsh(transform_covariance_eci_to_unw(P, R_LEO, V_LEO))
        assert np.all(eigvals > 0)
    run_test("cov stays positive definite", t_cov_positive_definite)

    def t_cov_cross_ecr_unw():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_unw_to_ecr(
                transform_covariance_ecr_to_unw(P, JD_TEST, R_LEO, V_LEO),
                JD_TEST, R_LEO, V_LEO), P, atol=1e-6)
    run_test("cov ECR↔UNW roundtrip", t_cov_cross_ecr_unw)

    def t_cov_unw_tnw():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_tnw_to_unw(
                transform_covariance_unw_to_tnw(P, R_LEO, V_LEO), R_LEO, V_LEO),
            P, atol=1e-6)
    run_test("cov UNW↔TNW roundtrip", t_cov_unw_tnw)

    def t_cov_ecr_tnw():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_tnw_to_ecr(
                transform_covariance_ecr_to_tnw(P, JD_TEST, R_LEO, V_LEO),
                JD_TEST, R_LEO, V_LEO), P, atol=1e-6)
    run_test("cov ECR↔TNW roundtrip", t_cov_ecr_tnw)

    def t_cov_invalid_shape():
        try:
            transform_covariance(np.eye(4), "eci", "unw", r_eci=R_LEO, v_eci=V_LEO)
            assert False, "Should raise"
        except ValueError:
            pass
    run_test("cov invalid shape → ValueError", t_cov_invalid_shape)

    # ── Unified API ──
    def t_unified_identity():
        npt.assert_allclose(get_dcm("eci", "eci"), np.eye(3), atol=1e-15)
    run_test("unified: get_dcm('eci','eci') = I", t_unified_identity)

    def t_unified_ecr():
        npt.assert_allclose(get_dcm("eci", "ecr", jd=JD_TEST), eci_to_ecr_matrix(JD_TEST), atol=1e-14)
    run_test("unified: get_dcm matches explicit", t_unified_ecr)

    def t_unified_transform():
        v = np.array([1e6, 2e6, 3e6])
        npt.assert_allclose(
            transform(v, "eci", "unw", r_eci=R_LEO, v_eci=V_LEO),
            eci_to_unw(v, R_LEO, V_LEO), atol=1e-8)
    run_test("unified: transform() matches explicit", t_unified_transform)

    def t_unified_full_chain():
        v = np.array([5e6, -3e6, 1e6])
        v1 = transform(v, "ecr", "unw", jd=JD_TEST, r_eci=R_LEO, v_eci=V_LEO)
        v2 = transform(v1, "unw", "tnw", r_eci=R_LEO, v_eci=V_LEO)
        v3 = transform(v2, "tnw", "eci", r_eci=R_LEO, v_eci=V_LEO)
        v4 = transform(v3, "eci", "ecr", jd=JD_TEST)
        npt.assert_allclose(v4, v, atol=1e-6)
    run_test("unified: ECR→UNW→TNW→ECI→ECR chain", t_unified_full_chain)

    def t_unified_bad_frame():
        try:
            get_dcm("eci", "xyz")
            assert False
        except ValueError:
            pass
    run_test("unified: invalid frame → ValueError", t_unified_bad_frame)

    def t_unified_missing_jd():
        try:
            get_dcm("eci", "ecr")  # no jd
            assert False
        except ValueError:
            pass
    run_test("unified: missing jd → ValueError", t_unified_missing_jd)

    def t_ecr_unw_via_eci():
        v = np.array([1e6, -5e5, 2e6])
        npt.assert_allclose(
            ecr_to_unw(v, JD_TEST, R_LEO, V_LEO),
            eci_to_unw(ecr_to_eci(v, JD_TEST), R_LEO, V_LEO), atol=1e-8)
    run_test("ECR→UNW = ECR→ECI→UNW consistency", t_ecr_unw_via_eci)

    def t_legacy_alias():
        npt.assert_allclose(eci_to_ecef(R_LEO, JD_TEST), eci_to_ecr(R_LEO, JD_TEST), atol=1e-10)
    run_test("legacy eci_to_ecef == eci_to_ecr", t_legacy_alias)


# ═══════════════════════════════════════════════════════════════════════════
#  3. ORBITS MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_orbits():
    print("\n── orbits ──")

    def t_kep_rt():
        a, e, i = R_EARTH + 600e3, 0.001, np.deg2rad(45)
        r, v = keplerian_to_eci(a, e, i, np.deg2rad(100), np.deg2rad(50), np.deg2rad(120))
        oe = eci_to_keplerian(r, v)
        npt.assert_allclose(oe["a"], a, rtol=1e-10)
        npt.assert_allclose(oe["e"], e, atol=1e-10)
        npt.assert_allclose(oe["i"], i, atol=1e-10)
    run_test("Keplerian ↔ ECI roundtrip (a, e, i)", t_kep_rt)

    def t_kep_rt_angles():
        raan, argp, nu = np.deg2rad(100), np.deg2rad(50), np.deg2rad(120)
        r, v = keplerian_to_eci(R_EARTH + 600e3, 0.01, np.deg2rad(45), raan, argp, nu)
        oe = eci_to_keplerian(r, v)
        npt.assert_allclose(oe["raan"], raan, atol=1e-8)
        npt.assert_allclose(oe["argp"], argp, atol=1e-8)
        npt.assert_allclose(oe["nu"], nu, atol=1e-8)
    run_test("Keplerian ↔ ECI roundtrip (RAAN, argp, ν)", t_kep_rt_angles)

    def t_energy():
        E0 = np.linalg.norm(V_LEO)**2/2 - MU_EARTH/np.linalg.norm(R_LEO)
        r1, v1 = propagate_kepler(R_LEO, V_LEO, 3600)
        E1 = np.linalg.norm(v1)**2/2 - MU_EARTH/np.linalg.norm(r1)
        npt.assert_allclose(E0, E1, rtol=1e-10)
    run_test("energy conservation (1 hour)", t_energy)

    def t_angular_momentum():
        h0 = np.cross(R_LEO, V_LEO)
        r1, v1 = propagate_kepler(R_LEO, V_LEO, 7200)
        npt.assert_allclose(np.cross(r1, v1), h0, rtol=1e-10)
    run_test("angular momentum conservation (2 hours)", t_angular_momentum)

    def t_period_return():
        oe = eci_to_keplerian(R_LEO, V_LEO)
        T = compute_orbital_period(oe["a"])
        r1, v1 = propagate_kepler(R_LEO, V_LEO, T)
        npt.assert_allclose(r1, R_LEO, rtol=1e-8)
        npt.assert_allclose(v1, V_LEO, rtol=1e-8)
    run_test("full period returns to start", t_period_return)

    def t_mean_motion():
        a = R_EARTH + 500e3
        n = compute_mean_motion(a)
        T = compute_orbital_period(a)
        npt.assert_allclose(n * T, 2 * np.pi, rtol=1e-12)
    run_test("mean_motion × period = 2π", t_mean_motion)

    def t_period_leo():
        T = compute_orbital_period(R_EARTH + 400e3)
        assert 5000 < T < 6000  # ~90 min
    run_test("LEO period ~90 min", t_period_leo)

    def t_period_geo():
        T = compute_orbital_period(42_164e3)
        npt.assert_allclose(T, 86164, rtol=0.01)  # sidereal day
    run_test("GEO period ~86164 s", t_period_geo)

    def t_j2_raan_drift():
        r0, v0 = keplerian_to_eci(R_EARTH + 500e3, 0.001, np.deg2rad(51.6), 0, 0, 0)
        oe0 = eci_to_keplerian(r0, v0)
        r1, v1 = propagate_j2(r0, v0, 86400)
        oe1 = eci_to_keplerian(r1, v1)
        # Handle 2π wrapping
        raan_drift = (oe1["raan"] - oe0["raan"]) % (2 * np.pi)
        if raan_drift > np.pi:
            raan_drift = 2 * np.pi - raan_drift
        assert 0.01 < raan_drift < 0.2, f"RAAN drift {np.rad2deg(raan_drift):.4f}°"
    run_test("J2 RAAN drift over 1 day", t_j2_raan_drift)

    def t_j2_sma_constant():
        r0, v0 = keplerian_to_eci(R_EARTH + 500e3, 0.001, np.deg2rad(51.6), 0, 0, 0)
        oe0 = eci_to_keplerian(r0, v0)
        r1, v1 = propagate_j2(r0, v0, 86400)
        oe1 = eci_to_keplerian(r1, v1)
        npt.assert_allclose(oe1["a"], oe0["a"], rtol=1e-6)
    run_test("J2 preserves semi-major axis", t_j2_sma_constant)

    def t_propagate_negative_dt():
        r1, v1 = propagate_kepler(R_LEO, V_LEO, 3600)
        r0_back, v0_back = propagate_kepler(r1, v1, -3600)
        npt.assert_allclose(r0_back, R_LEO, rtol=1e-7)
    run_test("propagate negative dt (backward)", t_propagate_negative_dt)

    def t_kep_circular():
        # Circular orbit: e ≈ 0
        r, v = keplerian_to_eci(R_EARTH + 500e3, 1e-8, np.deg2rad(45), 0, 0, np.deg2rad(90))
        oe = eci_to_keplerian(r, v)
        assert oe["e"] < 1e-5
    run_test("circular orbit e ≈ 0", t_kep_circular)


# ═══════════════════════════════════════════════════════════════════════════
#  4. COVERAGE MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_coverage():
    print("\n── coverage ──")

    def t_earth_hit_nadir():
        hit = earth_intersection(R_LEO, -normalize(R_LEO))
        assert hit is not None
        npt.assert_allclose(np.linalg.norm(hit), R_EARTH, rtol=1e-6)
    run_test("Earth intersection nadir", t_earth_hit_nadir)

    def t_earth_miss_tangent():
        assert earth_intersection(R_LEO, normalize(V_LEO)) is None
    run_test("Earth miss tangent direction", t_earth_miss_tangent)

    def t_earth_miss_outward():
        assert earth_intersection(R_LEO, normalize(R_LEO)) is None
    run_test("Earth miss outward direction", t_earth_miss_outward)

    def t_footprint_center():
        fp = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(15))
        assert fp["center_eci"] is not None
        npt.assert_allclose(np.linalg.norm(fp["center_eci"]), R_EARTH, rtol=1e-4)
    run_test("nadir footprint center on Earth", t_footprint_center)

    def t_footprint_valid_count():
        fp = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(10), n_points=72)
        assert np.sum(fp["valid"]) > 60
    run_test("footprint valid points > 60/72", t_footprint_valid_count)

    def t_footprint_tnw():
        # In TNW for near-circular orbit, nadir is approximately along -N (row 1)
        R_tnw = eci_to_tnw_matrix(R_LEO, V_LEO)
        nadir_eci = -normalize(R_LEO)
        nadir_tnw = R_tnw @ nadir_eci
        fp = sensor_footprint_tnw(R_LEO, V_LEO, nadir_tnw, np.deg2rad(15))
        assert fp["center_eci"] is not None
    run_test("TNW footprint with nadir boresight", t_footprint_tnw)

    def t_footprint_with_lla():
        fp = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(15), jd=JD_TEST)
        assert "lla" in fp and "center_lla" in fp
    run_test("footprint with JD returns LLA", t_footprint_with_lla)

    def t_footprint_ring_size():
        fp1 = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(5))
        fp2 = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(30))
        # Wider cone → more spread
        pts1 = fp1["eci"][fp1["valid"]]
        pts2 = fp2["eci"][fp2["valid"]]
        if len(pts1) > 2 and len(pts2) > 2:
            spread1 = np.std(pts1, axis=0)
            spread2 = np.std(pts2, axis=0)
            assert np.linalg.norm(spread2) > np.linalg.norm(spread1)
    run_test("wider cone → larger footprint", t_footprint_ring_size)

    def t_access_nadir():
        sub = normalize(R_LEO) * R_EARTH
        g = access_geometry(R_LEO, V_LEO, sub)
        npt.assert_allclose(g["elevation"], np.pi / 2, atol=0.01)
    run_test("access geometry nadir → 90° elevation", t_access_nadir)

    def t_access_keys():
        sub = normalize(R_LEO) * R_EARTH
        g = access_geometry(R_LEO, V_LEO, sub)
        for k in ["range", "los_eci", "los_unw", "los_tnw", "elevation", "azimuth_unw"]:
            assert k in g, f"missing key: {k}"
    run_test("access geometry all output keys", t_access_keys)

    def t_ground_trace():
        oe = eci_to_keplerian(R_LEO, V_LEO)
        T = compute_orbital_period(oe["a"])
        trace = ground_trace(R_LEO, V_LEO, T, n_points=100)
        assert trace.shape == (100, 3)
    run_test("ground trace shape", t_ground_trace)

    def t_ground_trace_lat_bound():
        oe = eci_to_keplerian(R_LEO, V_LEO)
        T = compute_orbital_period(oe["a"])
        trace = ground_trace(R_LEO, V_LEO, T, n_points=200)
        assert np.max(np.abs(trace[:, 0])) < np.deg2rad(55)
    run_test("ground trace latitude bounded by inclination", t_ground_trace_lat_bound)


# ═══════════════════════════════════════════════════════════════════════════
#  5. SUN MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_sun():
    print("\n── sun ──")

    def t_sun_dist():
        d = sun_distance(JD_TEST)
        assert 0.97 * AU < d < 1.03 * AU
    run_test("Sun distance ~1 AU", t_sun_dist)

    def t_sun_unit():
        npt.assert_allclose(np.linalg.norm(sun_direction_eci(JD_TEST)), 1.0, atol=1e-14)
    run_test("Sun direction is unit vector", t_sun_unit)

    def t_sun_pos_magnitude():
        npt.assert_allclose(np.linalg.norm(sun_position_eci(JD_TEST)), sun_distance(JD_TEST), atol=1.0)
    run_test("Sun position norm = sun_distance", t_sun_pos_magnitude)

    def t_sun_dec():
        _, dec = solar_declination_ra(JD_TEST)
        assert abs(dec) < np.deg2rad(24)
    run_test("Sun declination within ±24°", t_sun_dec)

    def t_sun_ra():
        ra, _ = solar_declination_ra(JD_TEST)
        assert 0 <= ra < 2 * np.pi
    run_test("Sun RA in [0, 2π)", t_sun_ra)

    def t_sun_seasonal():
        # Summer solstice: dec > 20°
        jd_summer = julian_date(2026, 6, 21, 12)
        _, dec = solar_declination_ra(jd_summer)
        assert dec > np.deg2rad(20)
        # Winter solstice: dec < -20°
        jd_winter = julian_date(2026, 12, 21, 12)
        _, dec = solar_declination_ra(jd_winter)
        assert dec < np.deg2rad(-20)
    run_test("Sun seasonal declination", t_sun_seasonal)

    def t_sun_angle_toward():
        s = sun_direction_eci(JD_TEST)
        npt.assert_allclose(sun_angle(R_LEO, s, JD_TEST), 0.0, atol=0.01)
    run_test("Sun angle = 0 when pointing at Sun", t_sun_angle_toward)

    def t_sun_angle_away():
        s = sun_direction_eci(JD_TEST)
        npt.assert_allclose(sun_angle(R_LEO, -s, JD_TEST), np.pi, atol=0.01)
    run_test("Sun angle = π when pointing away", t_sun_angle_away)

    def t_sun_angle_perp():
        s = sun_direction_eci(JD_TEST)
        perp = normalize(np.cross(s, [0, 0, 1]))
        npt.assert_allclose(sun_angle(R_LEO, perp, JD_TEST), np.pi / 2, atol=0.01)
    run_test("Sun angle = π/2 perpendicular", t_sun_angle_perp)

    def t_excl_toward():
        s = sun_direction_eci(JD_TEST)
        r = check_solar_exclusion(R_LEO, s, JD_TEST, np.deg2rad(30))
        assert r["excluded"] and r["margin"] < 0
    run_test("solar exclusion: toward Sun → excluded", t_excl_toward)

    def t_excl_away():
        s = sun_direction_eci(JD_TEST)
        r = check_solar_exclusion(R_LEO, -s, JD_TEST, np.deg2rad(30))
        assert not r["excluded"] and r["margin"] > 0
    run_test("solar exclusion: away → clear", t_excl_away)

    def t_eclipse_cyl_sunward():
        s = sun_direction_eci(JD_TEST)
        assert eclipse_cylindrical(s * (R_EARTH + 500e3), JD_TEST) == "sunlit"
    run_test("eclipse cyl: sunward = sunlit", t_eclipse_cyl_sunward)

    def t_eclipse_cyl_shadow():
        s = sun_direction_eci(JD_TEST)
        assert eclipse_cylindrical(-s * (R_EARTH + 500e3), JD_TEST) == "eclipse"
    run_test("eclipse cyl: anti-Sun = eclipse", t_eclipse_cyl_shadow)

    def t_eclipse_conical_sunlit():
        s = sun_direction_eci(JD_TEST)
        r = eclipse_conical(s * (R_EARTH + 500e3), JD_TEST)
        assert r["state"] == "sunlit" and r["shadow_fraction"] == 0.0
    run_test("eclipse conical: sunlit state", t_eclipse_conical_sunlit)

    def t_eclipse_conical_umbra():
        s = sun_direction_eci(JD_TEST)
        r = eclipse_conical(-s * (R_EARTH + 500e3), JD_TEST)
        assert r["state"] in ("umbra", "penumbra") and r["shadow_fraction"] > 0
    run_test("eclipse conical: shadow behind Earth", t_eclipse_conical_umbra)

    def t_phase_angle_range():
        p = solar_phase_angle(R_LEO, normalize(R_LEO) * R_EARTH, JD_TEST)
        assert 0 <= p <= np.pi
    run_test("phase angle in [0, π]", t_phase_angle_range)

    def t_sep():
        assert 0 <= sep_angle(R_LEO, JD_TEST) <= np.pi
    run_test("SEP angle in [0, π]", t_sep)

    def t_illuminated():
        s = sun_direction_eci(JD_TEST)
        assert is_target_illuminated(s * (R_EARTH + 500e3), JD_TEST) == True
    run_test("target illuminated on sunward side", t_illuminated)

    def t_not_illuminated():
        s = sun_direction_eci(JD_TEST)
        assert is_target_illuminated(-s * (R_EARTH + 500e3), JD_TEST) == False
    run_test("target not illuminated behind Earth", t_not_illuminated)

    def t_eclipse_intervals():
        ints = eclipse_intervals(R_LEO, V_LEO, 6000, dt_step=30, jd_epoch=JD_TEST)
        assert isinstance(ints, list)
        for iv in ints:
            assert iv["start"] < iv["end"]
            assert iv["duration"] > 0
    run_test("eclipse_intervals structure", t_eclipse_intervals)

    def t_solar_excl_windows():
        bore = lambda r, v: -normalize(r)
        wins = solar_exclusion_windows(R_LEO, V_LEO, bore, np.deg2rad(30),
                                        3600, dt_step=60, jd_epoch=JD_TEST)
        assert isinstance(wins, list)
    run_test("solar_exclusion_windows callable", t_solar_excl_windows)

    def t_sun_constants():
        assert AU > 1.49e11
        assert R_SUN > 6e8
        assert SOLAR_FLUX_1AU > 1300
    run_test("Sun constants sanity", t_sun_constants)


# ═══════════════════════════════════════════════════════════════════════════
#  6. MOON MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_moon():
    print("\n── moon ──")

    def t_moon_dist():
        d = moon_distance(JD_TEST)
        assert 0.9 * MEAN_EARTH_MOON_DIST < d < 1.1 * MEAN_EARTH_MOON_DIST
    run_test("Moon distance ~384,400 km", t_moon_dist)

    def t_moon_unit():
        npt.assert_allclose(np.linalg.norm(moon_direction_eci(JD_TEST)), 1.0, atol=1e-14)
    run_test("Moon direction is unit vector", t_moon_unit)

    def t_moon_pos_norm():
        npt.assert_allclose(np.linalg.norm(moon_position_eci(JD_TEST)), moon_distance(JD_TEST), atol=1.0)
    run_test("Moon position norm = moon_distance", t_moon_pos_norm)

    def t_moon_ang_rad():
        a = moon_angular_radius(JD_TEST)
        assert np.deg2rad(0.2) < a < np.deg2rad(0.35)
    run_test("Moon angular radius ~0.26°", t_moon_ang_rad)

    def t_moon_illum():
        f = moon_illumination_fraction(JD_TEST)
        assert 0.0 <= f <= 1.0
    run_test("Moon illumination in [0, 1]", t_moon_illum)

    def t_moon_age():
        a = moon_age_days(JD_TEST)
        assert 0 <= a <= 29.54
    run_test("Moon age in [0, 29.5] days", t_moon_age)

    def t_moon_phase_name():
        n = moon_phase_name(JD_TEST)
        valid = {"New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                 "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"}
        assert n in valid
    run_test("Moon phase name valid", t_moon_phase_name)

    def t_moon_full():
        jd_full = julian_date(2026, 3, 3, 12)
        assert moon_illumination_fraction(jd_full) > 0.85
    run_test("near-full Moon illumination > 0.85", t_moon_full)

    def t_moon_new():
        # New Moon ~ March 19, 2026
        jd_new = julian_date(2026, 3, 19, 12)
        assert moon_illumination_fraction(jd_new) < 0.15
    run_test("near-new Moon illumination < 0.15", t_moon_new)

    def t_lunar_excl_toward():
        m = moon_direction_eci(JD_TEST)
        r = check_lunar_exclusion(R_LEO, m, JD_TEST, np.deg2rad(10))
        assert r["excluded"]
        assert "moon_phase_fraction" in r
    run_test("lunar exclusion when pointing at Moon", t_lunar_excl_toward)

    def t_lunar_excl_away():
        m = moon_direction_eci(JD_TEST)
        r = check_lunar_exclusion(R_LEO, -m, JD_TEST, np.deg2rad(10))
        assert not r["excluded"]
    run_test("no lunar exclusion away from Moon", t_lunar_excl_away)

    def t_sun_moon_angle():
        a = sun_moon_angle(JD_TEST)
        assert 0 <= a <= np.pi
    run_test("Sun-Moon angle in [0, π]", t_sun_moon_angle)

    def t_sun_moon_not_same():
        s = sun_direction_eci(JD_TEST)
        m = moon_direction_eci(JD_TEST)
        assert np.arccos(np.clip(np.dot(s, m), -1, 1)) > np.deg2rad(1)
    run_test("Sun ≠ Moon direction", t_sun_moon_not_same)

    def t_lunar_excl_windows():
        bore = lambda r, v: -normalize(r)
        wins = lunar_exclusion_windows(R_LEO, V_LEO, bore, np.deg2rad(10),
                                        3600, dt_step=60, jd_epoch=JD_TEST)
        assert isinstance(wins, list)
    run_test("lunar_exclusion_windows callable", t_lunar_excl_windows)

    def t_moon_earth_shadow_callable():
        assert isinstance(moon_earth_shadow(R_LEO, JD_TEST), bool)
    run_test("moon_earth_shadow returns bool", t_moon_earth_shadow_callable)

    def t_moon_constants():
        assert R_MOON > 1.7e6
        assert MEAN_EARTH_MOON_DIST > 3.8e8
    run_test("Moon constants sanity", t_moon_constants)

    def t_moon_phase_angle_range():
        pa = moon_phase_angle(JD_TEST)
        assert 0 <= pa <= np.pi
    run_test("Moon phase angle in [0, π]", t_moon_phase_angle_range)


# ═══════════════════════════════════════════════════════════════════════════
#  7. EXCLUSION MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_exclusion():
    print("\n── exclusion ──")

    def t_clear():
        s = sun_direction_eci(JD_TEST); m = moon_direction_eci(JD_TEST)
        b = normalize(np.cross(s, m))
        r = check_all_exclusions(R_LEO, V_LEO, b, JD_TEST)
        assert not r["sun_excluded"] and not r["moon_excluded"]
    run_test("clear when pointing away from both", t_clear)

    def t_sun_violation():
        s = sun_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, s, JD_TEST)
        assert r["sun_excluded"] and not r["available"]
        assert "solar" in r["violations"]
    run_test("sun violation detected", t_sun_violation)

    def t_moon_violation():
        m = moon_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, m, JD_TEST)
        assert r["moon_excluded"] and "lunar" in r["violations"]
    run_test("moon violation detected", t_moon_violation)

    def t_eclipse_field():
        s = sun_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, -s, JD_TEST)
        assert "eclipse_state" in r and r["eclipse_state"] in ("sunlit", "penumbra", "umbra")
    run_test("eclipse state in output", t_eclipse_field)

    def t_output_keys():
        s = sun_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, -s, JD_TEST)
        for k in ["available", "sun_excluded", "moon_excluded", "earth_excluded",
                   "eclipse", "sun_angle", "moon_angle", "earth_angle", "violations"]:
            assert k in r, f"missing key: {k}"
    run_test("all exclusion output keys present", t_output_keys)

    def t_timeline():
        bore = lambda r, v: -normalize(r)
        tl = availability_timeline(R_LEO, V_LEO, bore, 3600, dt_step=60, jd_epoch=JD_TEST)
        assert len(tl["times"]) == 60
        assert len(tl["available"]) == 60
        assert 0 <= tl["duty_cycle"] <= 1.0
        for k in ["solar_exclusion_fraction", "lunar_exclusion_fraction",
                   "earth_limb_fraction", "eclipse_fraction", "total_unavailable_fraction"]:
            assert k in tl["exclusion_summary"]
    run_test("availability timeline structure + summary", t_timeline)

    def t_timeline_arrays():
        bore = lambda r, v: -normalize(r)
        tl = availability_timeline(R_LEO, V_LEO, bore, 600, dt_step=30, jd_epoch=JD_TEST)
        for k in ["sun_excluded", "moon_excluded", "earth_excluded", "eclipse",
                   "sun_angles", "moon_angles"]:
            assert k in tl and len(tl[k]) == len(tl["times"])
    run_test("availability timeline array lengths match", t_timeline_arrays)

    def t_target_obs():
        sub = normalize(R_LEO) * R_EARTH
        obs = target_observability(R_LEO, V_LEO, sub, JD_TEST,
                                   np.deg2rad(45), require_illuminated_target=False)
        assert obs["in_fov"] and obs["range"] > 0
        for k in ["observable", "sun_clear", "moon_clear", "sensor_sunlit",
                   "target_illuminated", "phase_angle", "reasons"]:
            assert k in obs
    run_test("target observability structure", t_target_obs)

    def t_target_obs_range():
        sub = normalize(R_LEO) * R_EARTH
        obs = target_observability(R_LEO, V_LEO, sub, JD_TEST, np.deg2rad(45),
                                   require_illuminated_target=False)
        expected_range = np.linalg.norm(R_LEO) - R_EARTH
        npt.assert_allclose(obs["range"], expected_range, rtol=0.01)
    run_test("target observability range correct", t_target_obs_range)


# ═══════════════════════════════════════════════════════════════════════════
#  8. TLE MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_tle():
    print("\n── tle ──")

    def t_parse():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        assert t.norad_id == 25544 and t.name == "ISS (ZARYA)"
        npt.assert_allclose(np.rad2deg(t.inclination), 51.64, atol=0.01)
        assert t.eccentricity == 0.0007
    run_test("TLE parse ISS (id, name, inc, ecc)", t_parse)

    def t_parse_fields():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        assert t.epoch_year == 2026
        npt.assert_allclose(t.epoch_day, 54.5, atol=0.01)
        npt.assert_allclose(np.rad2deg(t.raan), 210.0, atol=0.01)
        npt.assert_allclose(np.rad2deg(t.argp), 90.0, atol=0.01)
        npt.assert_allclose(np.rad2deg(t.mean_anomaly), 270.0, atol=0.01)
    run_test("TLE parsed field values", t_parse_fields)

    def t_parse_derived():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        assert t.semi_major_axis > R_EARTH
        assert 5000 < t.period < 6500
        assert t.epoch_jd > 2400000
        assert t.mean_motion > 0
    run_test("TLE derived quantities", t_parse_derived)

    def t_batch_3line():
        txt = f"{ISS_NAME}\n{ISS_L1}\n{ISS_L2}\n" * 3
        tles = parse_tle_batch(txt)
        assert len(tles) == 3 and all(t.norad_id == 25544 for t in tles)
    run_test("TLE batch parse (3-line format)", t_batch_3line)

    def t_batch_2line():
        txt = f"{ISS_L1}\n{ISS_L2}\n{ISS_L1}\n{ISS_L2}\n"
        tles = parse_tle_batch(txt)
        assert len(tles) == 2
    run_test("TLE batch parse (2-line format)", t_batch_2line)

    def t_batch_mixed():
        txt = f"{ISS_NAME}\n{ISS_L1}\n{ISS_L2}\n{ISS_L1}\n{ISS_L2}\n"
        tles = parse_tle_batch(txt)
        assert len(tles) == 2
    run_test("TLE batch mixed format", t_batch_mixed)

    def t_epoch_state_alt():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r, v = tle_epoch_state(t)
        alt = np.linalg.norm(r) - R_EARTH
        assert 350e3 < alt < 450e3
    run_test("TLE epoch state: ISS altitude 350-450 km", t_epoch_state_alt)

    def t_epoch_state_vel():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        _, v = tle_epoch_state(t)
        assert 7000 < np.linalg.norm(v) < 8000
    run_test("TLE epoch state: ISS velocity 7-8 km/s", t_epoch_state_vel)

    def t_propagate_1orbit():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r0, _ = tle_epoch_state(t)
        r1, _ = propagate_tle(t, t.epoch_jd + t.period / 86400)
        alt0, alt1 = np.linalg.norm(r0) - R_EARTH, np.linalg.norm(r1) - R_EARTH
        assert abs(alt1 - alt0) < 50e3
    run_test("TLE propagate 1 orbit altitude match", t_propagate_1orbit)

    def t_propagate_6h():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r, _ = propagate_tle(t, t.epoch_jd + 0.25)
        assert 300e3 < np.linalg.norm(r) - R_EARTH < 500e3
    run_test("TLE propagate 6h altitude valid", t_propagate_6h)

    def t_propagate_backward():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r, _ = propagate_tle(t, t.epoch_jd - 0.1)
        assert np.linalg.norm(r) > R_EARTH
    run_test("TLE propagate backward valid", t_propagate_backward)

    def t_dataclass_defaults():
        t = TLE()
        assert t.priority == 5 and t.revisit_rate == 0.0 and t.name == ""
    run_test("TLE dataclass defaults", t_dataclass_defaults)

    def t_multiple_tles():
        txt = '''ISS (ZARYA)
1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  9002
2 25544  51.6400 210.0000 0007000  90.0000 270.0000 15.50000000400000
CSS (TIANHE)
1 48274U 21035A   26054.50000000  .00020000  00000-0  22000-3 0  9001
2 48274  41.4700  50.0000 0005000 120.0000  60.0000 15.60000000100000'''
        tles = parse_tle_batch(txt)
        assert len(tles) == 2
        assert tles[0].norad_id == 25544 and tles[1].norad_id == 48274
    run_test("TLE multi-satellite batch", t_multiple_tles)

#!/usr/bin/env python3
"""
test_orbital_frames.py — Comprehensive Unit Test Suite
=======================================================

Tests all modules of the orbital_frames library:

  1. utils       — constants, vector ops, time, coordinate conversions
  2. frames      — ECI/ECR/UNW/TNW rotation matrices, transforms, covariance
  3. orbits      — Keplerian conversion, propagation, energy/momentum
  4. coverage    — footprints, ground trace, access geometry, coverage windows
  5. sun         — ephemeris, exclusion, eclipse, phase angle
  6. moon        — ephemeris, exclusion, illumination, phase
  7. exclusion   — combined exclusion, availability timeline, observability
  8. tle         — TLE parsing, epoch state, propagation
  9. sensor      — ground sensor model, azel, visibility, visual magnitude
 10. scheduler   — priority scoring, greedy scheduling, format output

Run:  python3 test_orbital_frames.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import numpy.testing as npt

from orbital_frames import *
from orbital_frames.frames import (
    state_eci_to_ecr, state_ecr_to_eci,
    ecr_to_unw_matrix, unw_to_ecr_matrix,
    ecr_to_tnw_matrix, tnw_to_ecr_matrix,
    _build_6x6,
)
from orbital_frames.orbits import propagate_j2
from orbital_frames.sun import eclipse_conical, eclipse_cylindrical, sep_angle
from orbital_frames.moon import (
    moon_phase_angle, moon_age_days, moon_phase_name,
    moon_angular_radius, sun_moon_angle, moon_earth_shadow,
)
from orbital_frames.exclusion import check_all_exclusions, target_observability
from orbital_frames.sensor import (
    GroundSensor, check_visibility, topocentric_azel,
    sun_elevation_at_sensor, estimate_visual_magnitude,
)
from orbital_frames.tle import TLE, parse_tle, parse_tle_batch, tle_epoch_state, propagate_tle
from orbital_frames.scheduler import (
    SatelliteTask, ScheduleResult,
    compute_visibility_windows, compute_urgency, compute_score,
    schedule_greedy, format_schedule,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════

JD_TEST = julian_date(2026, 2, 24, 12, 0, 0)

# Circular LEO (ISS-like)
_LEO = keplerian_to_eci(R_EARTH + 400e3, 0.0001, np.deg2rad(51.6),
                        np.deg2rad(30), 0.0, np.deg2rad(45))
R_LEO, V_LEO = _LEO

# Elliptical HEO (Molniya-like)
_HEO = keplerian_to_eci(26_600e3, 0.74, np.deg2rad(63.4), 0.0,
                        np.deg2rad(270), np.deg2rad(90))
R_HEO, V_HEO = _HEO

# Reference ISS TLE
ISS_NAME = "ISS (ZARYA)"
ISS_L1 = "1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  9002"
ISS_L2 = "2 25544  51.6400 210.0000 0007000  90.0000 270.0000 15.50000000400000"

# Reference sensors
SENSOR_RADAR = GroundSensor(
    name="TestRadar", lat=np.deg2rad(42.62), lon=np.deg2rad(-71.49),
    alt=146.0, sensor_type="radar", min_elevation=np.deg2rad(5.0),
    max_range=40_000e3,
)
SENSOR_OPTICAL = GroundSensor(
    name="TestOptical", lat=np.deg2rad(32.0), lon=np.deg2rad(-106.0),
    alt=1300.0, sensor_type="optical", limiting_magnitude=18.0,
    min_elevation=np.deg2rad(10.0),
    solar_exclusion_angle=np.deg2rad(30.0),
    lunar_exclusion_angle=np.deg2rad(10.0),
)


# ═══════════════════════════════════════════════════════════════════════════
#  Test Runner Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

_results = {"pass": 0, "fail": 0, "errors": []}


def run_test(name, fn):
    """Execute a single test, track results."""
    try:
        fn()
        print(f"  PASS  {name}")
        _results["pass"] += 1
    except Exception as ex:
        print(f"  FAIL  {name}")
        print(f"         {ex}")
        traceback.print_exc(limit=3)
        _results["fail"] += 1
        _results["errors"].append(name)


# ═══════════════════════════════════════════════════════════════════════════
#  1. UTILS MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_utils():
    print("\n── utils ──")

    def t_normalize_unit():
        v = np.array([3.0, 4.0, 0.0])
        npt.assert_allclose(np.linalg.norm(normalize(v)), 1.0, atol=1e-15)
    run_test("normalize → unit length", t_normalize_unit)

    def t_normalize_direction():
        v = np.array([3.0, 4.0, 0.0])
        npt.assert_allclose(normalize(v), [0.6, 0.8, 0.0], atol=1e-15)
    run_test("normalize direction correct", t_normalize_direction)

    def t_normalize_batch():
        vecs = np.random.randn(20, 3) * 100
        normed = normalize(vecs)
        npt.assert_allclose(np.linalg.norm(normed, axis=1), np.ones(20), atol=1e-14)
    run_test("normalize batch (N,3)", t_normalize_batch)

    def t_normalize_zero_raises():
        try:
            normalize(np.zeros(3))
            assert False, "Should raise"
        except ValueError:
            pass
    run_test("normalize zero → ValueError", t_normalize_zero_raises)

    def t_rotation_identity():
        R = rotation_matrix_axis_angle(np.array([0, 0, 1]), 0.0)
        npt.assert_allclose(R, np.eye(3), atol=1e-14)
    run_test("rotation 0° = identity", t_rotation_identity)

    def t_rotation_90z():
        R = rotation_matrix_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        npt.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-14)
    run_test("rotation 90° about Z: x̂→ŷ", t_rotation_90z)

    def t_rotation_180():
        R = rotation_matrix_axis_angle(np.array([0, 0, 1]), np.pi)
        npt.assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-14)
    run_test("rotation 180° about Z: x̂→−x̂", t_rotation_180)

    def t_rotation_orthogonal():
        R = rotation_matrix_axis_angle(np.array([1, 1, 1]), 1.23)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    run_test("rotation matrix orthogonal + det=1", t_rotation_orthogonal)

    def t_jd_j2000():
        npt.assert_allclose(julian_date(2000, 1, 1, 12), 2451545.0, atol=1e-6)
    run_test("JD at J2000.0 = 2451545.0", t_jd_j2000)

    def t_jd_known():
        jd = julian_date(2026, 2, 24, 0, 0, 0)
        assert abs(jd - 2461095.5) < 1.0
    run_test("JD known date sanity", t_jd_known)

    def t_jd_monotonic():
        jd1 = julian_date(2026, 1, 1, 0)
        jd2 = julian_date(2026, 6, 1, 0)
        assert jd2 > jd1
    run_test("JD monotonically increasing", t_jd_monotonic)

    def t_gmst_range():
        g = gmst(JD_TEST)
        assert 0 <= g < 2 * np.pi
    run_test("GMST in [0, 2π)", t_gmst_range)

    def t_gmst_increases():
        g1 = gmst(JD_TEST)
        g2 = gmst(JD_TEST + 0.01)  # ~15 min later
        # GMST should change (Earth rotates)
        assert abs(g2 - g1) > 0
    run_test("GMST changes with time", t_gmst_increases)

    def t_eci_ecef_roundtrip():
        r_ecef = eci_to_ecef(R_LEO, JD_TEST)
        r_back = ecef_to_eci(r_ecef, JD_TEST)
        npt.assert_allclose(r_back, R_LEO, atol=1e-6)
    run_test("ECI↔ECEF roundtrip", t_eci_ecef_roundtrip)

    def t_eci_ecef_magnitude():
        r_ecef = eci_to_ecef(R_LEO, JD_TEST)
        npt.assert_allclose(np.linalg.norm(r_ecef), np.linalg.norm(R_LEO), atol=1e-8)
    run_test("ECEF preserves magnitude", t_eci_ecef_magnitude)

    def t_eci_ecef_batch():
        vecs = np.random.randn(30, 3) * R_EARTH
        ecef = eci_to_ecef(vecs, JD_TEST)
        back = ecef_to_eci(ecef, JD_TEST)
        npt.assert_allclose(back, vecs, atol=1e-6)
    run_test("ECI↔ECEF batch (30,3)", t_eci_ecef_batch)

    def t_lla_equator():
        r = lla_to_ecef(0.0, 0.0, 0.0)
        npt.assert_allclose(r[0], R_EARTH, rtol=1e-10)
        npt.assert_allclose(r[1], 0.0, atol=1e-6)
        npt.assert_allclose(r[2], 0.0, atol=1e-6)
    run_test("LLA equator → ECEF x-axis", t_lla_equator)

    def t_lla_pole():
        r = lla_to_ecef(np.pi / 2, 0.0, 0.0)
        npt.assert_allclose(r[0], 0.0, atol=1e-3)
        npt.assert_allclose(r[1], 0.0, atol=1e-3)
        assert r[2] > 0
    run_test("LLA north pole → +Z", t_lla_pole)

    def t_lla_roundtrip():
        lat, lon, alt = np.deg2rad(40), np.deg2rad(-74), 100.0
        r = lla_to_ecef(lat, lon, alt)
        lla = ecef_to_lla(r)
        npt.assert_allclose(lla[0], lat, atol=1e-10)
        npt.assert_allclose(lla[1], lon, atol=1e-10)
        npt.assert_allclose(lla[2], alt, atol=0.01)
    run_test("LLA↔ECEF roundtrip", t_lla_roundtrip)

    def t_lla_roundtrip_batch():
        lats = np.deg2rad(np.array([-80, -30, 0, 30, 80]))
        lons = np.deg2rad(np.array([0, 45, 90, 135, 180]))
        pts = np.array([lla_to_ecef(la, lo, 5000.0) for la, lo in zip(lats, lons)])
        lla = ecef_to_lla(pts)
        npt.assert_allclose(lla[:, 0], lats, atol=1e-9)
        npt.assert_allclose(lla[:, 1], lons, atol=1e-9)
    run_test("LLA↔ECEF batch roundtrip", t_lla_roundtrip_batch)

    def t_lla_altitude():
        alt = 50000.0  # 50 km
        r = lla_to_ecef(np.deg2rad(45), np.deg2rad(90), alt)
        lla = ecef_to_lla(r)
        npt.assert_allclose(lla[2], alt, atol=0.1)
    run_test("LLA altitude preserved", t_lla_altitude)

    def t_constants():
        assert MU_EARTH > 3.9e14
        assert R_EARTH > 6.37e6
        assert 0 < J2 < 0.01
        assert OMEGA_EARTH > 7e-5
        assert 0 < F_EARTH < 0.01
        assert 0 < E2_EARTH < 0.01
    run_test("physical constants sanity", t_constants)


# ═══════════════════════════════════════════════════════════════════════════
#  2. FRAMES MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_frames():
    print("\n── frames ──")

    # ── UNW ──
    def t_unw_orthogonal():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    run_test("UNW DCM orthogonal + det=1", t_unw_orthogonal)

    def t_unw_u_radial():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[0], normalize(R_LEO), atol=1e-14)
    run_test("UNW U-axis = radial", t_unw_u_radial)

    def t_unw_w_orbit_normal():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        h = np.cross(R_LEO, V_LEO)
        npt.assert_allclose(R[2], normalize(h), atol=1e-14)
    run_test("UNW W-axis = orbit normal", t_unw_w_orbit_normal)

    def t_unw_n_completes_rhs():
        R = eci_to_unw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[1], np.cross(R[2], R[0]), atol=1e-14)
    run_test("UNW N = W×U (right-hand)", t_unw_n_completes_rhs)

    def t_unw_elliptical():
        R = eci_to_unw_matrix(R_HEO, V_HEO)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    run_test("UNW orthogonal (elliptical orbit)", t_unw_elliptical)

    # ── TNW ──
    def t_tnw_orthogonal():
        R = eci_to_tnw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    run_test("TNW DCM orthogonal + det=1", t_tnw_orthogonal)

    def t_tnw_t_velocity():
        R = eci_to_tnw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[0], normalize(V_LEO), atol=1e-14)
    run_test("TNW T-axis = velocity", t_tnw_t_velocity)

    def t_tnw_n_completes():
        R = eci_to_tnw_matrix(R_LEO, V_LEO)
        npt.assert_allclose(R[1], np.cross(R[2], R[0]), atol=1e-14)
    run_test("TNW N = W×T (right-hand)", t_tnw_n_completes)

    def t_w_axes_match():
        npt.assert_allclose(
            eci_to_unw_matrix(R_LEO, V_LEO)[2],
            eci_to_tnw_matrix(R_LEO, V_LEO)[2], atol=1e-14)
    run_test("UNW.W == TNW.W (shared orbit normal)", t_w_axes_match)

    def t_unw_tnw_diverge_eccentric():
        # For eccentric orbits at ν≠0,180, UNW and TNW should differ
        R_unw = eci_to_unw_matrix(R_HEO, V_HEO)
        R_tnw = eci_to_tnw_matrix(R_HEO, V_HEO)
        # T and N_unw should NOT be identical
        cos_ang = np.clip(np.dot(R_tnw[0], R_unw[1]), -1, 1)
        angle = np.arccos(cos_ang)
        assert angle > np.deg2rad(5), f"UNW/TNW divergence only {np.rad2deg(angle):.1f}°"
    run_test("UNW/TNW diverge for eccentric orbit", t_unw_tnw_diverge_eccentric)

    # ── ECR ──
    def t_ecr_orthogonal():
        R = eci_to_ecr_matrix(JD_TEST)
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    run_test("ECR DCM orthogonal", t_ecr_orthogonal)

    def t_ecr_z_invariant():
        R = eci_to_ecr_matrix(JD_TEST)
        npt.assert_allclose(R[2], [0, 0, 1], atol=1e-14)
    run_test("ECR Z-axis invariant (pole unchanged)", t_ecr_z_invariant)

    def t_ecr_inverse():
        R = eci_to_ecr_matrix(JD_TEST)
        npt.assert_allclose(R @ ecr_to_eci_matrix(JD_TEST), np.eye(3), atol=1e-14)
    run_test("ECR R·R⁻¹ = I", t_ecr_inverse)

    # ── Vector roundtrips ──
    def t_eci_unw_rt():
        v = np.array([1e3, -2e3, 3e3])
        npt.assert_allclose(unw_to_eci(eci_to_unw(v, R_LEO, V_LEO), R_LEO, V_LEO), v, atol=1e-8)
    run_test("ECI→UNW→ECI roundtrip", t_eci_unw_rt)

    def t_eci_tnw_rt():
        v = np.array([1e3, -2e3, 3e3])
        npt.assert_allclose(tnw_to_eci(eci_to_tnw(v, R_LEO, V_LEO), R_LEO, V_LEO), v, atol=1e-8)
    run_test("ECI→TNW→ECI roundtrip", t_eci_tnw_rt)

    def t_unw_tnw_rt():
        v = np.array([100, 200, 300])
        npt.assert_allclose(tnw_to_unw(unw_to_tnw(v, R_LEO, V_LEO), R_LEO, V_LEO), v, atol=1e-8)
    run_test("UNW→TNW→UNW roundtrip", t_unw_tnw_rt)

    def t_eci_ecr_rt():
        npt.assert_allclose(ecr_to_eci(eci_to_ecr(R_LEO, JD_TEST), JD_TEST), R_LEO, atol=1e-8)
    run_test("ECI→ECR→ECI roundtrip", t_eci_ecr_rt)

    def t_ecr_unw_rt():
        v = np.array([1e6, 2e6, 3e6])
        npt.assert_allclose(
            unw_to_ecr(ecr_to_unw(v, JD_TEST, R_LEO, V_LEO), JD_TEST, R_LEO, V_LEO),
            v, atol=1e-6)
    run_test("ECR→UNW→ECR roundtrip", t_ecr_unw_rt)

    def t_ecr_tnw_rt():
        v = np.array([1e6, 2e6, 3e6])
        npt.assert_allclose(
            tnw_to_ecr(ecr_to_tnw(v, JD_TEST, R_LEO, V_LEO), JD_TEST, R_LEO, V_LEO),
            v, atol=1e-6)
    run_test("ECR→TNW→ECR roundtrip", t_ecr_tnw_rt)

    def t_batch_transform():
        vecs = np.random.randn(50, 3) * 1000
        npt.assert_allclose(unw_to_eci(eci_to_unw(vecs, R_LEO, V_LEO), R_LEO, V_LEO), vecs, atol=1e-8)
    run_test("batch (50,3) UNW roundtrip", t_batch_transform)

    def t_magnitude_preserved():
        v = np.array([5e3, -3e3, 7e3])
        npt.assert_allclose(np.linalg.norm(eci_to_unw(v, R_LEO, V_LEO)), np.linalg.norm(v), atol=1e-8)
    run_test("vector magnitude preserved", t_magnitude_preserved)

    # ── State transforms (transport theorem) ──
    def t_state_ecr_rt():
        r_ecr, v_ecr = state_eci_to_ecr(R_LEO, V_LEO, JD_TEST)
        r_back, v_back = state_ecr_to_eci(r_ecr, v_ecr, JD_TEST)
        npt.assert_allclose(r_back, R_LEO, atol=1e-8)
        npt.assert_allclose(v_back, V_LEO, atol=1e-8)
    run_test("state ECI↔ECR roundtrip (transport theorem)", t_state_ecr_rt)

    def t_transport_coriolis():
        v_ecr_wrong = eci_to_ecr(V_LEO, JD_TEST)
        _, v_ecr_right = state_eci_to_ecr(R_LEO, V_LEO, JD_TEST)
        diff = np.linalg.norm(v_ecr_right - v_ecr_wrong)
        assert 1.0 < diff < 600, f"Coriolis correction {diff:.1f} m/s"
    run_test("transport theorem Coriolis non-trivial", t_transport_coriolis)

    def t_ground_station_zero_ecr_vel():
        r_gs_ecef = lla_to_ecef(np.deg2rad(40), np.deg2rad(-74), 0)
        r_gs_eci = ecef_to_eci(r_gs_ecef, JD_TEST)
        omega = np.array([0, 0, OMEGA_EARTH])
        v_gs_eci = np.cross(omega, r_gs_eci)
        _, v_ecr = state_eci_to_ecr(r_gs_eci, v_gs_eci, JD_TEST)
        npt.assert_allclose(v_ecr, np.zeros(3), atol=1e-4)
    run_test("ground station → zero ECR velocity", t_ground_station_zero_ecr_vel)

    def t_state_unw():
        state = np.concatenate([R_LEO, V_LEO]) + np.array([100, 200, 300, 1, 2, 3], dtype=float)
        s_unw = state_eci_to_unw(state, R_LEO, V_LEO)
        assert s_unw.shape == (6,)
    run_test("state_eci_to_unw 6-vector", t_state_unw)

    # ── Covariance ──
    def t_cov3_unw_rt():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_unw_to_eci(transform_covariance_eci_to_unw(P, R_LEO, V_LEO), R_LEO, V_LEO),
            P, atol=1e-6)
    run_test("cov 3×3 ECI↔UNW roundtrip", t_cov3_unw_rt)

    def t_cov6_tnw_rt():
        A = np.random.randn(6, 6) * 100; P = A @ A.T + np.eye(6) * 10
        npt.assert_allclose(
            transform_covariance_tnw_to_eci(transform_covariance_eci_to_tnw(P, R_LEO, V_LEO), R_LEO, V_LEO),
            P, atol=1e-4)
    run_test("cov 6×6 ECI↔TNW roundtrip", t_cov6_tnw_rt)

    def t_cov_ecr_rt():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_ecr_to_eci(transform_covariance_eci_to_ecr(P, JD_TEST), JD_TEST),
            P, atol=1e-6)
    run_test("cov 3×3 ECI↔ECR roundtrip", t_cov_ecr_rt)

    def t_cov_trace():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(np.trace(P), np.trace(transform_covariance_eci_to_unw(P, R_LEO, V_LEO)), rtol=1e-12)
    run_test("cov trace preserved under rotation", t_cov_trace)

    def t_cov_positive_definite():
        A = np.random.randn(3, 3) * 50; P = A @ A.T + np.eye(3) * 5
        eigvals = np.linalg.eigvalsh(transform_covariance_eci_to_unw(P, R_LEO, V_LEO))
        assert np.all(eigvals > 0)
    run_test("cov stays positive definite", t_cov_positive_definite)

    def t_cov_cross_ecr_unw():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_unw_to_ecr(
                transform_covariance_ecr_to_unw(P, JD_TEST, R_LEO, V_LEO),
                JD_TEST, R_LEO, V_LEO), P, atol=1e-6)
    run_test("cov ECR↔UNW roundtrip", t_cov_cross_ecr_unw)

    def t_cov_unw_tnw():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_tnw_to_unw(
                transform_covariance_unw_to_tnw(P, R_LEO, V_LEO), R_LEO, V_LEO),
            P, atol=1e-6)
    run_test("cov UNW↔TNW roundtrip", t_cov_unw_tnw)

    def t_cov_ecr_tnw():
        A = np.random.randn(3, 3) * 100; P = A @ A.T + np.eye(3) * 10
        npt.assert_allclose(
            transform_covariance_tnw_to_ecr(
                transform_covariance_ecr_to_tnw(P, JD_TEST, R_LEO, V_LEO),
                JD_TEST, R_LEO, V_LEO), P, atol=1e-6)
    run_test("cov ECR↔TNW roundtrip", t_cov_ecr_tnw)

    def t_cov_invalid_shape():
        try:
            transform_covariance(np.eye(4), "eci", "unw", r_eci=R_LEO, v_eci=V_LEO)
            assert False, "Should raise"
        except ValueError:
            pass
    run_test("cov invalid shape → ValueError", t_cov_invalid_shape)

    # ── Unified API ──
    def t_unified_identity():
        npt.assert_allclose(get_dcm("eci", "eci"), np.eye(3), atol=1e-15)
    run_test("unified: get_dcm('eci','eci') = I", t_unified_identity)

    def t_unified_ecr():
        npt.assert_allclose(get_dcm("eci", "ecr", jd=JD_TEST), eci_to_ecr_matrix(JD_TEST), atol=1e-14)
    run_test("unified: get_dcm matches explicit", t_unified_ecr)

    def t_unified_transform():
        v = np.array([1e6, 2e6, 3e6])
        npt.assert_allclose(
            transform(v, "eci", "unw", r_eci=R_LEO, v_eci=V_LEO),
            eci_to_unw(v, R_LEO, V_LEO), atol=1e-8)
    run_test("unified: transform() matches explicit", t_unified_transform)

    def t_unified_full_chain():
        v = np.array([5e6, -3e6, 1e6])
        v1 = transform(v, "ecr", "unw", jd=JD_TEST, r_eci=R_LEO, v_eci=V_LEO)
        v2 = transform(v1, "unw", "tnw", r_eci=R_LEO, v_eci=V_LEO)
        v3 = transform(v2, "tnw", "eci", r_eci=R_LEO, v_eci=V_LEO)
        v4 = transform(v3, "eci", "ecr", jd=JD_TEST)
        npt.assert_allclose(v4, v, atol=1e-6)
    run_test("unified: ECR→UNW→TNW→ECI→ECR chain", t_unified_full_chain)

    def t_unified_bad_frame():
        try:
            get_dcm("eci", "xyz")
            assert False
        except ValueError:
            pass
    run_test("unified: invalid frame → ValueError", t_unified_bad_frame)

    def t_unified_missing_jd():
        try:
            get_dcm("eci", "ecr")  # no jd
            assert False
        except ValueError:
            pass
    run_test("unified: missing jd → ValueError", t_unified_missing_jd)

    def t_ecr_unw_via_eci():
        v = np.array([1e6, -5e5, 2e6])
        npt.assert_allclose(
            ecr_to_unw(v, JD_TEST, R_LEO, V_LEO),
            eci_to_unw(ecr_to_eci(v, JD_TEST), R_LEO, V_LEO), atol=1e-8)
    run_test("ECR→UNW = ECR→ECI→UNW consistency", t_ecr_unw_via_eci)

    def t_legacy_alias():
        npt.assert_allclose(eci_to_ecef(R_LEO, JD_TEST), eci_to_ecr(R_LEO, JD_TEST), atol=1e-10)
    run_test("legacy eci_to_ecef == eci_to_ecr", t_legacy_alias)


# ═══════════════════════════════════════════════════════════════════════════
#  3. ORBITS MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_orbits():
    print("\n── orbits ──")

    def t_kep_rt():
        a, e, i = R_EARTH + 600e3, 0.001, np.deg2rad(45)
        r, v = keplerian_to_eci(a, e, i, np.deg2rad(100), np.deg2rad(50), np.deg2rad(120))
        oe = eci_to_keplerian(r, v)
        npt.assert_allclose(oe["a"], a, rtol=1e-10)
        npt.assert_allclose(oe["e"], e, atol=1e-10)
        npt.assert_allclose(oe["i"], i, atol=1e-10)
    run_test("Keplerian ↔ ECI roundtrip (a, e, i)", t_kep_rt)

    def t_kep_rt_angles():
        raan, argp, nu = np.deg2rad(100), np.deg2rad(50), np.deg2rad(120)
        r, v = keplerian_to_eci(R_EARTH + 600e3, 0.01, np.deg2rad(45), raan, argp, nu)
        oe = eci_to_keplerian(r, v)
        npt.assert_allclose(oe["raan"], raan, atol=1e-8)
        npt.assert_allclose(oe["argp"], argp, atol=1e-8)
        npt.assert_allclose(oe["nu"], nu, atol=1e-8)
    run_test("Keplerian ↔ ECI roundtrip (RAAN, argp, ν)", t_kep_rt_angles)

    def t_energy():
        E0 = np.linalg.norm(V_LEO)**2/2 - MU_EARTH/np.linalg.norm(R_LEO)
        r1, v1 = propagate_kepler(R_LEO, V_LEO, 3600)
        E1 = np.linalg.norm(v1)**2/2 - MU_EARTH/np.linalg.norm(r1)
        npt.assert_allclose(E0, E1, rtol=1e-10)
    run_test("energy conservation (1 hour)", t_energy)

    def t_angular_momentum():
        h0 = np.cross(R_LEO, V_LEO)
        r1, v1 = propagate_kepler(R_LEO, V_LEO, 7200)
        npt.assert_allclose(np.cross(r1, v1), h0, rtol=1e-10)
    run_test("angular momentum conservation (2 hours)", t_angular_momentum)

    def t_period_return():
        oe = eci_to_keplerian(R_LEO, V_LEO)
        T = compute_orbital_period(oe["a"])
        r1, v1 = propagate_kepler(R_LEO, V_LEO, T)
        npt.assert_allclose(r1, R_LEO, rtol=1e-8)
        npt.assert_allclose(v1, V_LEO, rtol=1e-8)
    run_test("full period returns to start", t_period_return)

    def t_mean_motion():
        a = R_EARTH + 500e3
        n = compute_mean_motion(a)
        T = compute_orbital_period(a)
        npt.assert_allclose(n * T, 2 * np.pi, rtol=1e-12)
    run_test("mean_motion × period = 2π", t_mean_motion)

    def t_period_leo():
        T = compute_orbital_period(R_EARTH + 400e3)
        assert 5000 < T < 6000  # ~90 min
    run_test("LEO period ~90 min", t_period_leo)

    def t_period_geo():
        T = compute_orbital_period(42_164e3)
        npt.assert_allclose(T, 86164, rtol=0.01)  # sidereal day
    run_test("GEO period ~86164 s", t_period_geo)

    def t_j2_raan_drift():
        r0, v0 = keplerian_to_eci(R_EARTH + 500e3, 0.001, np.deg2rad(51.6), 0, 0, 0)
        oe0 = eci_to_keplerian(r0, v0)
        r1, v1 = propagate_j2(r0, v0, 86400)
        oe1 = eci_to_keplerian(r1, v1)
        # Handle 2π wrapping
        raan_drift = (oe1["raan"] - oe0["raan"]) % (2 * np.pi)
        if raan_drift > np.pi:
            raan_drift = 2 * np.pi - raan_drift
        assert 0.01 < raan_drift < 0.2, f"RAAN drift {np.rad2deg(raan_drift):.4f}°"
    run_test("J2 RAAN drift over 1 day", t_j2_raan_drift)

    def t_j2_sma_constant():
        r0, v0 = keplerian_to_eci(R_EARTH + 500e3, 0.001, np.deg2rad(51.6), 0, 0, 0)
        oe0 = eci_to_keplerian(r0, v0)
        r1, v1 = propagate_j2(r0, v0, 86400)
        oe1 = eci_to_keplerian(r1, v1)
        npt.assert_allclose(oe1["a"], oe0["a"], rtol=1e-6)
    run_test("J2 preserves semi-major axis", t_j2_sma_constant)

    def t_propagate_negative_dt():
        r1, v1 = propagate_kepler(R_LEO, V_LEO, 3600)
        r0_back, v0_back = propagate_kepler(r1, v1, -3600)
        npt.assert_allclose(r0_back, R_LEO, rtol=1e-7)
    run_test("propagate negative dt (backward)", t_propagate_negative_dt)

    def t_kep_circular():
        # Circular orbit: e ≈ 0
        r, v = keplerian_to_eci(R_EARTH + 500e3, 1e-8, np.deg2rad(45), 0, 0, np.deg2rad(90))
        oe = eci_to_keplerian(r, v)
        assert oe["e"] < 1e-5
    run_test("circular orbit e ≈ 0", t_kep_circular)


# ═══════════════════════════════════════════════════════════════════════════
#  4. COVERAGE MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_coverage():
    print("\n── coverage ──")

    def t_earth_hit_nadir():
        hit = earth_intersection(R_LEO, -normalize(R_LEO))
        assert hit is not None
        npt.assert_allclose(np.linalg.norm(hit), R_EARTH, rtol=1e-6)
    run_test("Earth intersection nadir", t_earth_hit_nadir)

    def t_earth_miss_tangent():
        assert earth_intersection(R_LEO, normalize(V_LEO)) is None
    run_test("Earth miss tangent direction", t_earth_miss_tangent)

    def t_earth_miss_outward():
        assert earth_intersection(R_LEO, normalize(R_LEO)) is None
    run_test("Earth miss outward direction", t_earth_miss_outward)

    def t_footprint_center():
        fp = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(15))
        assert fp["center_eci"] is not None
        npt.assert_allclose(np.linalg.norm(fp["center_eci"]), R_EARTH, rtol=1e-4)
    run_test("nadir footprint center on Earth", t_footprint_center)

    def t_footprint_valid_count():
        fp = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(10), n_points=72)
        assert np.sum(fp["valid"]) > 60
    run_test("footprint valid points > 60/72", t_footprint_valid_count)

    def t_footprint_tnw():
        # In TNW for near-circular orbit, nadir is approximately along -N (row 1)
        R_tnw = eci_to_tnw_matrix(R_LEO, V_LEO)
        nadir_eci = -normalize(R_LEO)
        nadir_tnw = R_tnw @ nadir_eci
        fp = sensor_footprint_tnw(R_LEO, V_LEO, nadir_tnw, np.deg2rad(15))
        assert fp["center_eci"] is not None
    run_test("TNW footprint with nadir boresight", t_footprint_tnw)

    def t_footprint_with_lla():
        fp = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(15), jd=JD_TEST)
        assert "lla" in fp and "center_lla" in fp
    run_test("footprint with JD returns LLA", t_footprint_with_lla)

    def t_footprint_ring_size():
        fp1 = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(5))
        fp2 = sensor_footprint_unw(R_LEO, V_LEO, [-1, 0, 0], np.deg2rad(30))
        # Wider cone → more spread
        pts1 = fp1["eci"][fp1["valid"]]
        pts2 = fp2["eci"][fp2["valid"]]
        if len(pts1) > 2 and len(pts2) > 2:
            spread1 = np.std(pts1, axis=0)
            spread2 = np.std(pts2, axis=0)
            assert np.linalg.norm(spread2) > np.linalg.norm(spread1)
    run_test("wider cone → larger footprint", t_footprint_ring_size)

    def t_access_nadir():
        sub = normalize(R_LEO) * R_EARTH
        g = access_geometry(R_LEO, V_LEO, sub)
        npt.assert_allclose(g["elevation"], np.pi / 2, atol=0.01)
    run_test("access geometry nadir → 90° elevation", t_access_nadir)

    def t_access_keys():
        sub = normalize(R_LEO) * R_EARTH
        g = access_geometry(R_LEO, V_LEO, sub)
        for k in ["range", "los_eci", "los_unw", "los_tnw", "elevation", "azimuth_unw"]:
            assert k in g, f"missing key: {k}"
    run_test("access geometry all output keys", t_access_keys)

    def t_ground_trace():
        oe = eci_to_keplerian(R_LEO, V_LEO)
        T = compute_orbital_period(oe["a"])
        trace = ground_trace(R_LEO, V_LEO, T, n_points=100)
        assert trace.shape == (100, 3)
    run_test("ground trace shape", t_ground_trace)

    def t_ground_trace_lat_bound():
        oe = eci_to_keplerian(R_LEO, V_LEO)
        T = compute_orbital_period(oe["a"])
        trace = ground_trace(R_LEO, V_LEO, T, n_points=200)
        assert np.max(np.abs(trace[:, 0])) < np.deg2rad(55)
    run_test("ground trace latitude bounded by inclination", t_ground_trace_lat_bound)


# ═══════════════════════════════════════════════════════════════════════════
#  5. SUN MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_sun():
    print("\n── sun ──")

    def t_sun_dist():
        d = sun_distance(JD_TEST)
        assert 0.97 * AU < d < 1.03 * AU
    run_test("Sun distance ~1 AU", t_sun_dist)

    def t_sun_unit():
        npt.assert_allclose(np.linalg.norm(sun_direction_eci(JD_TEST)), 1.0, atol=1e-14)
    run_test("Sun direction is unit vector", t_sun_unit)

    def t_sun_pos_magnitude():
        npt.assert_allclose(np.linalg.norm(sun_position_eci(JD_TEST)), sun_distance(JD_TEST), atol=1.0)
    run_test("Sun position norm = sun_distance", t_sun_pos_magnitude)

    def t_sun_dec():
        _, dec = solar_declination_ra(JD_TEST)
        assert abs(dec) < np.deg2rad(24)
    run_test("Sun declination within ±24°", t_sun_dec)

    def t_sun_ra():
        ra, _ = solar_declination_ra(JD_TEST)
        assert 0 <= ra < 2 * np.pi
    run_test("Sun RA in [0, 2π)", t_sun_ra)

    def t_sun_seasonal():
        # Summer solstice: dec > 20°
        jd_summer = julian_date(2026, 6, 21, 12)
        _, dec = solar_declination_ra(jd_summer)
        assert dec > np.deg2rad(20)
        # Winter solstice: dec < -20°
        jd_winter = julian_date(2026, 12, 21, 12)
        _, dec = solar_declination_ra(jd_winter)
        assert dec < np.deg2rad(-20)
    run_test("Sun seasonal declination", t_sun_seasonal)

    def t_sun_angle_toward():
        s = sun_direction_eci(JD_TEST)
        npt.assert_allclose(sun_angle(R_LEO, s, JD_TEST), 0.0, atol=0.01)
    run_test("Sun angle = 0 when pointing at Sun", t_sun_angle_toward)

    def t_sun_angle_away():
        s = sun_direction_eci(JD_TEST)
        npt.assert_allclose(sun_angle(R_LEO, -s, JD_TEST), np.pi, atol=0.01)
    run_test("Sun angle = π when pointing away", t_sun_angle_away)

    def t_sun_angle_perp():
        s = sun_direction_eci(JD_TEST)
        perp = normalize(np.cross(s, [0, 0, 1]))
        npt.assert_allclose(sun_angle(R_LEO, perp, JD_TEST), np.pi / 2, atol=0.01)
    run_test("Sun angle = π/2 perpendicular", t_sun_angle_perp)

    def t_excl_toward():
        s = sun_direction_eci(JD_TEST)
        r = check_solar_exclusion(R_LEO, s, JD_TEST, np.deg2rad(30))
        assert r["excluded"] and r["margin"] < 0
    run_test("solar exclusion: toward Sun → excluded", t_excl_toward)

    def t_excl_away():
        s = sun_direction_eci(JD_TEST)
        r = check_solar_exclusion(R_LEO, -s, JD_TEST, np.deg2rad(30))
        assert not r["excluded"] and r["margin"] > 0
    run_test("solar exclusion: away → clear", t_excl_away)

    def t_eclipse_cyl_sunward():
        s = sun_direction_eci(JD_TEST)
        assert eclipse_cylindrical(s * (R_EARTH + 500e3), JD_TEST) == "sunlit"
    run_test("eclipse cyl: sunward = sunlit", t_eclipse_cyl_sunward)

    def t_eclipse_cyl_shadow():
        s = sun_direction_eci(JD_TEST)
        assert eclipse_cylindrical(-s * (R_EARTH + 500e3), JD_TEST) == "eclipse"
    run_test("eclipse cyl: anti-Sun = eclipse", t_eclipse_cyl_shadow)

    def t_eclipse_conical_sunlit():
        s = sun_direction_eci(JD_TEST)
        r = eclipse_conical(s * (R_EARTH + 500e3), JD_TEST)
        assert r["state"] == "sunlit" and r["shadow_fraction"] == 0.0
    run_test("eclipse conical: sunlit state", t_eclipse_conical_sunlit)

    def t_eclipse_conical_umbra():
        s = sun_direction_eci(JD_TEST)
        r = eclipse_conical(-s * (R_EARTH + 500e3), JD_TEST)
        assert r["state"] in ("umbra", "penumbra") and r["shadow_fraction"] > 0
    run_test("eclipse conical: shadow behind Earth", t_eclipse_conical_umbra)

    def t_phase_angle_range():
        p = solar_phase_angle(R_LEO, normalize(R_LEO) * R_EARTH, JD_TEST)
        assert 0 <= p <= np.pi
    run_test("phase angle in [0, π]", t_phase_angle_range)

    def t_sep():
        assert 0 <= sep_angle(R_LEO, JD_TEST) <= np.pi
    run_test("SEP angle in [0, π]", t_sep)

    def t_illuminated():
        s = sun_direction_eci(JD_TEST)
        assert is_target_illuminated(s * (R_EARTH + 500e3), JD_TEST) == True
    run_test("target illuminated on sunward side", t_illuminated)

    def t_not_illuminated():
        s = sun_direction_eci(JD_TEST)
        assert is_target_illuminated(-s * (R_EARTH + 500e3), JD_TEST) == False
    run_test("target not illuminated behind Earth", t_not_illuminated)

    def t_eclipse_intervals():
        ints = eclipse_intervals(R_LEO, V_LEO, 6000, dt_step=30, jd_epoch=JD_TEST)
        assert isinstance(ints, list)
        for iv in ints:
            assert iv["start"] < iv["end"]
            assert iv["duration"] > 0
    run_test("eclipse_intervals structure", t_eclipse_intervals)

    def t_solar_excl_windows():
        bore = lambda r, v: -normalize(r)
        wins = solar_exclusion_windows(R_LEO, V_LEO, bore, np.deg2rad(30),
                                        3600, dt_step=60, jd_epoch=JD_TEST)
        assert isinstance(wins, list)
    run_test("solar_exclusion_windows callable", t_solar_excl_windows)

    def t_sun_constants():
        assert AU > 1.49e11
        assert R_SUN > 6e8
        assert SOLAR_FLUX_1AU > 1300
    run_test("Sun constants sanity", t_sun_constants)


# ═══════════════════════════════════════════════════════════════════════════
#  6. MOON MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_moon():
    print("\n── moon ──")

    def t_moon_dist():
        d = moon_distance(JD_TEST)
        assert 0.9 * MEAN_EARTH_MOON_DIST < d < 1.1 * MEAN_EARTH_MOON_DIST
    run_test("Moon distance ~384,400 km", t_moon_dist)

    def t_moon_unit():
        npt.assert_allclose(np.linalg.norm(moon_direction_eci(JD_TEST)), 1.0, atol=1e-14)
    run_test("Moon direction is unit vector", t_moon_unit)

    def t_moon_pos_norm():
        npt.assert_allclose(np.linalg.norm(moon_position_eci(JD_TEST)), moon_distance(JD_TEST), atol=1.0)
    run_test("Moon position norm = moon_distance", t_moon_pos_norm)

    def t_moon_ang_rad():
        a = moon_angular_radius(JD_TEST)
        assert np.deg2rad(0.2) < a < np.deg2rad(0.35)
    run_test("Moon angular radius ~0.26°", t_moon_ang_rad)

    def t_moon_illum():
        f = moon_illumination_fraction(JD_TEST)
        assert 0.0 <= f <= 1.0
    run_test("Moon illumination in [0, 1]", t_moon_illum)

    def t_moon_age():
        a = moon_age_days(JD_TEST)
        assert 0 <= a <= 29.54
    run_test("Moon age in [0, 29.5] days", t_moon_age)

    def t_moon_phase_name():
        n = moon_phase_name(JD_TEST)
        valid = {"New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                 "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"}
        assert n in valid
    run_test("Moon phase name valid", t_moon_phase_name)

    def t_moon_full():
        jd_full = julian_date(2026, 3, 3, 12)
        assert moon_illumination_fraction(jd_full) > 0.85
    run_test("near-full Moon illumination > 0.85", t_moon_full)

    def t_moon_new():
        # New Moon ~ March 19, 2026
        jd_new = julian_date(2026, 3, 19, 12)
        assert moon_illumination_fraction(jd_new) < 0.15
    run_test("near-new Moon illumination < 0.15", t_moon_new)

    def t_lunar_excl_toward():
        m = moon_direction_eci(JD_TEST)
        r = check_lunar_exclusion(R_LEO, m, JD_TEST, np.deg2rad(10))
        assert r["excluded"]
        assert "moon_phase_fraction" in r
    run_test("lunar exclusion when pointing at Moon", t_lunar_excl_toward)

    def t_lunar_excl_away():
        m = moon_direction_eci(JD_TEST)
        r = check_lunar_exclusion(R_LEO, -m, JD_TEST, np.deg2rad(10))
        assert not r["excluded"]
    run_test("no lunar exclusion away from Moon", t_lunar_excl_away)

    def t_sun_moon_angle():
        a = sun_moon_angle(JD_TEST)
        assert 0 <= a <= np.pi
    run_test("Sun-Moon angle in [0, π]", t_sun_moon_angle)

    def t_sun_moon_not_same():
        s = sun_direction_eci(JD_TEST)
        m = moon_direction_eci(JD_TEST)
        assert np.arccos(np.clip(np.dot(s, m), -1, 1)) > np.deg2rad(1)
    run_test("Sun ≠ Moon direction", t_sun_moon_not_same)

    def t_lunar_excl_windows():
        bore = lambda r, v: -normalize(r)
        wins = lunar_exclusion_windows(R_LEO, V_LEO, bore, np.deg2rad(10),
                                        3600, dt_step=60, jd_epoch=JD_TEST)
        assert isinstance(wins, list)
    run_test("lunar_exclusion_windows callable", t_lunar_excl_windows)

    def t_moon_earth_shadow_callable():
        assert isinstance(moon_earth_shadow(R_LEO, JD_TEST), bool)
    run_test("moon_earth_shadow returns bool", t_moon_earth_shadow_callable)

    def t_moon_constants():
        assert R_MOON > 1.7e6
        assert MEAN_EARTH_MOON_DIST > 3.8e8
    run_test("Moon constants sanity", t_moon_constants)

    def t_moon_phase_angle_range():
        pa = moon_phase_angle(JD_TEST)
        assert 0 <= pa <= np.pi
    run_test("Moon phase angle in [0, π]", t_moon_phase_angle_range)


# ═══════════════════════════════════════════════════════════════════════════
#  7. EXCLUSION MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_exclusion():
    print("\n── exclusion ──")

    def t_clear():
        s = sun_direction_eci(JD_TEST); m = moon_direction_eci(JD_TEST)
        b = normalize(np.cross(s, m))
        r = check_all_exclusions(R_LEO, V_LEO, b, JD_TEST)
        assert not r["sun_excluded"] and not r["moon_excluded"]
    run_test("clear when pointing away from both", t_clear)

    def t_sun_violation():
        s = sun_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, s, JD_TEST)
        assert r["sun_excluded"] and not r["available"]
        assert "solar" in r["violations"]
    run_test("sun violation detected", t_sun_violation)

    def t_moon_violation():
        m = moon_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, m, JD_TEST)
        assert r["moon_excluded"] and "lunar" in r["violations"]
    run_test("moon violation detected", t_moon_violation)

    def t_eclipse_field():
        s = sun_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, -s, JD_TEST)
        assert "eclipse_state" in r and r["eclipse_state"] in ("sunlit", "penumbra", "umbra")
    run_test("eclipse state in output", t_eclipse_field)

    def t_output_keys():
        s = sun_direction_eci(JD_TEST)
        r = check_all_exclusions(R_LEO, V_LEO, -s, JD_TEST)
        for k in ["available", "sun_excluded", "moon_excluded", "earth_excluded",
                   "eclipse", "sun_angle", "moon_angle", "earth_angle", "violations"]:
            assert k in r, f"missing key: {k}"
    run_test("all exclusion output keys present", t_output_keys)

    def t_timeline():
        bore = lambda r, v: -normalize(r)
        tl = availability_timeline(R_LEO, V_LEO, bore, 3600, dt_step=60, jd_epoch=JD_TEST)
        assert len(tl["times"]) == 60
        assert len(tl["available"]) == 60
        assert 0 <= tl["duty_cycle"] <= 1.0
        for k in ["solar_exclusion_fraction", "lunar_exclusion_fraction",
                   "earth_limb_fraction", "eclipse_fraction", "total_unavailable_fraction"]:
            assert k in tl["exclusion_summary"]
    run_test("availability timeline structure + summary", t_timeline)

    def t_timeline_arrays():
        bore = lambda r, v: -normalize(r)
        tl = availability_timeline(R_LEO, V_LEO, bore, 600, dt_step=30, jd_epoch=JD_TEST)
        for k in ["sun_excluded", "moon_excluded", "earth_excluded", "eclipse",
                   "sun_angles", "moon_angles"]:
            assert k in tl and len(tl[k]) == len(tl["times"])
    run_test("availability timeline array lengths match", t_timeline_arrays)

    def t_target_obs():
        sub = normalize(R_LEO) * R_EARTH
        obs = target_observability(R_LEO, V_LEO, sub, JD_TEST,
                                   np.deg2rad(45), require_illuminated_target=False)
        assert obs["in_fov"] and obs["range"] > 0
        for k in ["observable", "sun_clear", "moon_clear", "sensor_sunlit",
                   "target_illuminated", "phase_angle", "reasons"]:
            assert k in obs
    run_test("target observability structure", t_target_obs)

    def t_target_obs_range():
        sub = normalize(R_LEO) * R_EARTH
        obs = target_observability(R_LEO, V_LEO, sub, JD_TEST, np.deg2rad(45),
                                   require_illuminated_target=False)
        expected_range = np.linalg.norm(R_LEO) - R_EARTH
        npt.assert_allclose(obs["range"], expected_range, rtol=0.01)
    run_test("target observability range correct", t_target_obs_range)


# ═══════════════════════════════════════════════════════════════════════════
#  8. TLE MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_tle():
    print("\n── tle ──")

    def t_parse():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        assert t.norad_id == 25544 and t.name == "ISS (ZARYA)"
        npt.assert_allclose(np.rad2deg(t.inclination), 51.64, atol=0.01)
        assert t.eccentricity == 0.0007
    run_test("TLE parse ISS (id, name, inc, ecc)", t_parse)

    def t_parse_fields():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        assert t.epoch_year == 2026
        npt.assert_allclose(t.epoch_day, 54.5, atol=0.01)
        npt.assert_allclose(np.rad2deg(t.raan), 210.0, atol=0.01)
        npt.assert_allclose(np.rad2deg(t.argp), 90.0, atol=0.01)
        npt.assert_allclose(np.rad2deg(t.mean_anomaly), 270.0, atol=0.01)
    run_test("TLE parsed field values", t_parse_fields)

    def t_parse_derived():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        assert t.semi_major_axis > R_EARTH
        assert 5000 < t.period < 6500
        assert t.epoch_jd > 2400000
        assert t.mean_motion > 0
    run_test("TLE derived quantities", t_parse_derived)

    def t_batch_3line():
        txt = f"{ISS_NAME}\n{ISS_L1}\n{ISS_L2}\n" * 3
        tles = parse_tle_batch(txt)
        assert len(tles) == 3 and all(t.norad_id == 25544 for t in tles)
    run_test("TLE batch parse (3-line format)", t_batch_3line)

    def t_batch_2line():
        txt = f"{ISS_L1}\n{ISS_L2}\n{ISS_L1}\n{ISS_L2}\n"
        tles = parse_tle_batch(txt)
        assert len(tles) == 2
    run_test("TLE batch parse (2-line format)", t_batch_2line)

    def t_batch_mixed():
        txt = f"{ISS_NAME}\n{ISS_L1}\n{ISS_L2}\n{ISS_L1}\n{ISS_L2}\n"
        tles = parse_tle_batch(txt)
        assert len(tles) == 2
    run_test("TLE batch mixed format", t_batch_mixed)

    def t_epoch_state_alt():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r, v = tle_epoch_state(t)
        alt = np.linalg.norm(r) - R_EARTH
        assert 350e3 < alt < 450e3
    run_test("TLE epoch state: ISS altitude 350-450 km", t_epoch_state_alt)

    def t_epoch_state_vel():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        _, v = tle_epoch_state(t)
        assert 7000 < np.linalg.norm(v) < 8000
    run_test("TLE epoch state: ISS velocity 7-8 km/s", t_epoch_state_vel)

    def t_propagate_1orbit():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r0, _ = tle_epoch_state(t)
        r1, _ = propagate_tle(t, t.epoch_jd + t.period / 86400)
        alt0, alt1 = np.linalg.norm(r0) - R_EARTH, np.linalg.norm(r1) - R_EARTH
        assert abs(alt1 - alt0) < 50e3
    run_test("TLE propagate 1 orbit altitude match", t_propagate_1orbit)

    def t_propagate_6h():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r, _ = propagate_tle(t, t.epoch_jd + 0.25)
        assert 300e3 < np.linalg.norm(r) - R_EARTH < 500e3
    run_test("TLE propagate 6h altitude valid", t_propagate_6h)

    def t_propagate_backward():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r, _ = propagate_tle(t, t.epoch_jd - 0.1)
        assert np.linalg.norm(r) > R_EARTH
    run_test("TLE propagate backward valid", t_propagate_backward)

    def t_dataclass_defaults():
        t = TLE()
        assert t.priority == 5 and t.revisit_rate == 0.0 and t.name == ""
    run_test("TLE dataclass defaults", t_dataclass_defaults)

    def t_multiple_tles():
        txt = '''ISS (ZARYA)
1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  9002
2 25544  51.6400 210.0000 0007000  90.0000 270.0000 15.50000000400000
CSS (TIANHE)
1 48274U 21035A   26054.50000000  .00020000  00000-0  22000-3 0  9001
2 48274  41.4700  50.0000 0005000 120.0000  60.0000 15.60000000100000'''
        tles = parse_tle_batch(txt)
        assert len(tles) == 2
        assert tles[0].norad_id == 25544 and tles[1].norad_id == 48274
    run_test("TLE multi-satellite batch", t_multiple_tles)

    # ── Checksum ──
    from orbital_frames.tle import (
        tle_checksum, verify_checksum, verify_tle,
        tle_to_lines, tle_to_string,
        update_epoch, update_mean_anomaly,
    )

    def t_checksum_compute():
        # Known: digits sum, '-' counts as 1, all else 0
        assert tle_checksum("1 00000  00000-0  0") == (1 + 1) % 10
    run_test("TLE checksum: basic computation", t_checksum_compute)

    def t_checksum_digits():
        # All digits 0-9 should sum to 45, mod 10 = 5
        line = "1234567890" + " " * 58
        assert tle_checksum(line) == 5
    run_test("TLE checksum: digit sum", t_checksum_digits)

    def t_checksum_minus():
        # '-' contributes 1
        line = "---" + " " * 65
        assert tle_checksum(line) == 3
    run_test("TLE checksum: minus sign counts as 1", t_checksum_minus)

    def t_checksum_letters_ignored():
        line = "ABCXYZ" + " " * 62
        assert tle_checksum(line) == 0
    run_test("TLE checksum: letters contribute 0", t_checksum_letters_ignored)

    def t_verify_checksum_valid():
        # Build a line with correct checksum
        body = "1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  900"
        ck = tle_checksum(body)
        full_line = body + str(ck)
        assert verify_checksum(full_line)
    run_test("TLE verify_checksum: valid line", t_verify_checksum_valid)

    def t_verify_checksum_invalid():
        body = "1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  900"
        ck = tle_checksum(body)
        wrong = (ck + 1) % 10
        bad_line = body + str(wrong)
        assert not verify_checksum(bad_line)
    run_test("TLE verify_checksum: invalid line", t_verify_checksum_invalid)

    def t_verify_checksum_short():
        assert not verify_checksum("too short")
    run_test("TLE verify_checksum: short line → False", t_verify_checksum_short)

    def t_verify_tle_pair():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t)
        result = verify_tle(l1, l2)
        assert result["valid"], f"errors: {result['errors']}"
        assert result["line1_checksum"]
        assert result["line2_checksum"]
        assert result["norad_match"]
    run_test("TLE verify_tle: exported pair valid", t_verify_tle_pair)

    def t_verify_tle_norad_mismatch():
        l1 = "1 25544U 98067A   26054.50000000  .00016717  00000-0  10270-3 0  9002"
        l2 = "2 99999  51.6400 210.0000 0007000  90.0000 270.0000 15.50000000400000"
        l2 = l2[:68] + str(tle_checksum(l2))
        result = verify_tle(l1, l2)
        assert not result["norad_match"]
    run_test("TLE verify_tle: NORAD mismatch detected", t_verify_tle_norad_mismatch)

    # ── Export ──

    def t_export_lines_length():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t)
        assert len(l1) == 69, f"line1 len={len(l1)}"
        assert len(l2) == 69, f"line2 len={len(l2)}"
    run_test("TLE export: 69-char lines", t_export_lines_length)

    def t_export_lines_prefix():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t)
        assert l1.startswith("1 ")
        assert l2.startswith("2 ")
    run_test("TLE export: correct line prefixes", t_export_lines_prefix)

    def t_export_checksums_valid():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t)
        assert verify_checksum(l1), f"L1 checksum fail"
        assert verify_checksum(l2), f"L2 checksum fail"
    run_test("TLE export: checksums valid", t_export_checksums_valid)

    def t_export_norad_id():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t)
        assert int(l1[2:7]) == 25544
        assert int(l2[2:7]) == 25544
    run_test("TLE export: NORAD ID preserved", t_export_norad_id)

    def t_export_roundtrip_elements():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t1)
        t2 = parse_tle("", l1, l2)
        npt.assert_allclose(t2.inclination, t1.inclination, atol=1e-4)
        npt.assert_allclose(t2.raan, t1.raan, atol=1e-4)
        npt.assert_allclose(t2.eccentricity, t1.eccentricity, atol=1e-7)
        npt.assert_allclose(t2.argp, t1.argp, atol=1e-4)
        npt.assert_allclose(t2.mean_anomaly, t1.mean_anomaly, atol=1e-4)
        npt.assert_allclose(t2.mean_motion, t1.mean_motion, rtol=1e-6)
    run_test("TLE export→parse roundtrip preserves elements", t_export_roundtrip_elements)

    def t_export_roundtrip_epoch():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        l1, l2 = tle_to_lines(t1)
        t2 = parse_tle("", l1, l2)
        npt.assert_allclose(t2.epoch_jd, t1.epoch_jd, atol=1e-4)
    run_test("TLE export→parse roundtrip preserves epoch", t_export_roundtrip_epoch)

    def t_export_to_string_3line():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        txt = tle_to_string(t, include_name=True)
        lines = txt.strip().splitlines()
        assert len(lines) == 3
        assert lines[0] == "ISS (ZARYA)"
    run_test("TLE tle_to_string: 3-line with name", t_export_to_string_3line)

    def t_export_to_string_2line():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        txt = tle_to_string(t, include_name=False)
        lines = txt.strip().splitlines()
        assert len(lines) == 2
    run_test("TLE tle_to_string: 2-line without name", t_export_to_string_2line)

    # ── Update epoch ──

    def t_update_epoch_jd():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        new_jd = t1.epoch_jd + 1.0  # advance 1 day
        t2 = update_epoch(t1, new_jd)
        npt.assert_allclose(t2.epoch_jd, new_jd, atol=1e-6)
        assert t2.epoch_year > 0 and t2.epoch_day > 0
    run_test("TLE update_epoch: JD updated", t_update_epoch_jd)

    def t_update_epoch_immutable():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        orig_jd = t1.epoch_jd
        orig_ma = t1.mean_anomaly
        t2 = update_epoch(t1, t1.epoch_jd + 1.0)
        # Original should be unchanged
        assert t1.epoch_jd == orig_jd
        assert t1.mean_anomaly == orig_ma
    run_test("TLE update_epoch: original unchanged", t_update_epoch_immutable)

    def t_update_epoch_ma_advances():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_epoch(t1, t1.epoch_jd + 0.5)  # 12 hours
        # Mean anomaly should change (ISS does ~8 orbits in 12h)
        assert t2.mean_anomaly != t1.mean_anomaly
    run_test("TLE update_epoch: mean anomaly advances", t_update_epoch_ma_advances)

    def t_update_epoch_raan_drifts():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_epoch(t1, t1.epoch_jd + 1.0)
        # RAAN should drift due to J2
        assert t2.raan != t1.raan
    run_test("TLE update_epoch: RAAN drifts (J2)", t_update_epoch_raan_drifts)

    def t_update_epoch_no_propagate():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_epoch(t1, t1.epoch_jd + 1.0, propagate=False)
        # Only epoch changed; elements identical
        assert t2.epoch_jd != t1.epoch_jd
        assert t2.mean_anomaly == t1.mean_anomaly
        assert t2.raan == t1.raan
    run_test("TLE update_epoch: propagate=False preserves elements", t_update_epoch_no_propagate)

    def t_update_epoch_backward():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_epoch(t1, t1.epoch_jd - 0.5)
        assert t2.epoch_jd < t1.epoch_jd
        assert t2.mean_anomaly != t1.mean_anomaly
    run_test("TLE update_epoch: backward propagation", t_update_epoch_backward)

    def t_update_epoch_consistent_with_propagate():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        new_jd = t1.epoch_jd + 0.1  # ~2.4 hours
        # Propagate TLE to new_jd
        r_prop, v_prop = propagate_tle(t1, new_jd)
        # Update epoch and get state at the new epoch
        t2 = update_epoch(t1, new_jd)
        r_epoch, v_epoch = tle_epoch_state(t2)
        # Should agree within a few km (not exact due to mean vs osculating)
        npt.assert_allclose(r_epoch, r_prop, atol=50e3)
    run_test("TLE update_epoch: consistent with propagate_tle", t_update_epoch_consistent_with_propagate)

    def t_update_epoch_export_valid():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_epoch(t1, t1.epoch_jd + 1.0)
        l1, l2 = tle_to_lines(t2)
        assert verify_checksum(l1) and verify_checksum(l2)
    run_test("TLE update_epoch → export has valid checksums", t_update_epoch_export_valid)

    def t_update_epoch_revnum():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_epoch(t1, t1.epoch_jd + 1.0)  # ~15.5 revs/day
        assert t2.rev_number > t1.rev_number
    run_test("TLE update_epoch: rev number advances", t_update_epoch_revnum)

    def t_update_mean_anomaly():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_mean_anomaly(t1, np.deg2rad(90))
        npt.assert_allclose(np.rad2deg(t2.mean_anomaly), 90.0, atol=1e-10)
        # Original unchanged
        assert t1.mean_anomaly != t2.mean_anomaly
    run_test("TLE update_mean_anomaly: value set", t_update_mean_anomaly)

    def t_update_mean_anomaly_wraps():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = update_mean_anomaly(t1, np.deg2rad(370))  # wraps to 10°
        npt.assert_allclose(np.rad2deg(t2.mean_anomaly), 10.0, atol=1e-8)
    run_test("TLE update_mean_anomaly: wraps at 2π", t_update_mean_anomaly_wraps)


# ═══════════════════════════════════════════════════════════════════════════
#  9. SENSOR MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_sensor():
    print("\n── sensor ──")

    def t_sensor_ecef():
        r = SENSOR_RADAR.ecef()
        assert np.linalg.norm(r) > R_EARTH * 0.99
        assert len(r) == 3
    run_test("sensor ECEF position valid", t_sensor_ecef)

    def t_sensor_eci():
        r = SENSOR_RADAR.eci(JD_TEST)
        npt.assert_allclose(np.linalg.norm(r), np.linalg.norm(SENSOR_RADAR.ecef()), rtol=1e-10)
    run_test("sensor ECI magnitude matches ECEF", t_sensor_eci)

    def t_sensor_velocity():
        v = SENSOR_RADAR.eci_velocity(JD_TEST)
        assert 200 < np.linalg.norm(v) < 500
    run_test("sensor ECI velocity ~300-500 m/s", t_sensor_velocity)

    def t_sensor_equator_vel():
        s = GroundSensor(lat=0.0, lon=0.0)
        v = s.eci_velocity(JD_TEST)
        assert 400 < np.linalg.norm(v) < 500
    run_test("equatorial sensor velocity ~465 m/s", t_sensor_equator_vel)

    def t_azel_zenith():
        r_ecef = lla_to_ecef(0.0, 0.0, 0.0)
        r_target = lla_to_ecef(0.0, 0.0, 500e3)
        az, el, rng = topocentric_azel(r_ecef, 0.0, 0.0, r_target)
        npt.assert_allclose(el, np.pi / 2, atol=0.01)
        npt.assert_allclose(rng, 500e3, rtol=0.01)
    run_test("topocentric azel: zenith pass", t_azel_zenith)

    def t_azel_directional():
        # Verify azimuth convention: target due east at same lat
        lat_s, lon_s = np.deg2rad(40), np.deg2rad(0)
        r_sensor = lla_to_ecef(lat_s, lon_s, 0)
        r_east = lla_to_ecef(lat_s, np.deg2rad(5), 400e3)
        az_e, el_e, _ = topocentric_azel(r_sensor, lat_s, lon_s, r_east)
        # East target: verify az is consistent and el > 0
        assert el_e > 0
        # Verify different directions give different azimuths
        r_west = lla_to_ecef(lat_s, np.deg2rad(-5), 400e3)
        az_w, _, _ = topocentric_azel(r_sensor, lat_s, lon_s, r_west)
        assert abs(az_e - az_w) > np.deg2rad(90), "east/west should differ by >90°"
    run_test("topocentric azel: directional consistency", t_azel_directional)

    def t_azel_east():
        r_ecef = lla_to_ecef(0.0, 0.0, 0.0)
        r_target = lla_to_ecef(0.0, np.deg2rad(5), 500e3)
        az, el, rng = topocentric_azel(r_ecef, 0.0, 0.0, r_target)
        npt.assert_allclose(az, np.pi / 2, atol=np.deg2rad(15))
    run_test("topocentric azel: east azimuth", t_azel_east)

    def t_sun_elevation():
        el = sun_elevation_at_sensor(SENSOR_RADAR, JD_TEST)
        assert -np.pi / 2 <= el <= np.pi / 2
    run_test("sun elevation at sensor in bounds", t_sun_elevation)

    def t_vmag_finite():
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 1000e3), JD_TEST)
        r_sensor = GroundSensor(lat=0.0, lon=0.0).eci(JD_TEST)
        m = estimate_visual_magnitude(r_sat, r_sensor, JD_TEST, rcs=1.0)
        assert np.isfinite(m) and -5 < m < 30
    run_test("visual magnitude finite and reasonable", t_vmag_finite)

    def t_vmag_closer_brighter():
        r_s = GroundSensor(lat=0.0, lon=0.0).eci(JD_TEST)
        r1 = ecef_to_eci(lla_to_ecef(0.0, 0.0, 500e3), JD_TEST)
        r2 = ecef_to_eci(lla_to_ecef(0.0, 0.0, 2000e3), JD_TEST)
        m1 = estimate_visual_magnitude(r1, r_s, JD_TEST, rcs=1.0)
        m2 = estimate_visual_magnitude(r2, r_s, JD_TEST, rcs=1.0)
        assert m1 < m2
    run_test("closer satellite → brighter (lower mag)", t_vmag_closer_brighter)

    def t_vmag_larger_brighter():
        r_s = GroundSensor(lat=0.0, lon=0.0).eci(JD_TEST)
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 1000e3), JD_TEST)
        assert estimate_visual_magnitude(r_sat, r_s, JD_TEST, rcs=10.0) < \
               estimate_visual_magnitude(r_sat, r_s, JD_TEST, rcs=0.1)
    run_test("larger RCS → brighter", t_vmag_larger_brighter)

    def t_vis_radar_keys():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r_sat = propagate_tle(t, JD_TEST)[0]
        vis = check_visibility(SENSOR_RADAR, r_sat, JD_TEST)
        for k in ["visible", "el", "az", "range", "in_elevation", "in_azimuth",
                   "in_range", "in_for", "reasons"]:
            assert k in vis, f"missing key: {k}"
    run_test("radar visibility: all output keys", t_vis_radar_keys)

    def t_vis_optical_keys():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r_sat = propagate_tle(t, JD_TEST)[0]
        vis = check_visibility(SENSOR_OPTICAL, r_sat, JD_TEST, rcs=400.0)
        for k in ["sky_dark", "sat_illuminated", "sun_clear", "moon_clear",
                   "visual_mag", "mag_detectable"]:
            assert k in vis, f"missing key: {k}"
    run_test("optical visibility: all output keys", t_vis_optical_keys)

    def t_vis_below_horizon():
        r_sat = -normalize(SENSOR_RADAR.eci(JD_TEST)) * (R_EARTH + 400e3)
        vis = check_visibility(SENSOR_RADAR, r_sat, JD_TEST)
        assert not vis["visible"] and "elevation" in vis["reasons"]
    run_test("satellite below horizon → not visible", t_vis_below_horizon)

    def t_vis_out_of_range():
        s = GroundSensor(name="Short", lat=0.0, lon=0.0, sensor_type="radar",
                         max_range=100e3, min_elevation=np.deg2rad(-90))
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 500e3), JD_TEST)
        vis = check_visibility(s, r_sat, JD_TEST)
        assert not vis["visible"] and "range" in vis["reasons"]
    run_test("satellite out of range → not visible", t_vis_out_of_range)

    def t_sensor_for():
        s = GroundSensor(lat=0.0, lon=0.0, sensor_type="radar",
                         field_of_regard=np.deg2rad(30), min_elevation=np.deg2rad(0))
        # Zenith target should be in FOR (zenith angle = 0 < 30°)
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 500e3), JD_TEST)
        vis = check_visibility(s, r_sat, JD_TEST)
        assert vis["in_for"]
    run_test("field of regard: zenith within 30° FOR", t_sensor_for)


# ═══════════════════════════════════════════════════════════════════════════
#  10. SCHEDULER MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_scheduler():
    print("\n── scheduler ──")

    def t_urgency_fresh():
        npt.assert_allclose(compute_urgency(0, 3600), 1.0)
    run_test("urgency: fresh = 1.0", t_urgency_fresh)

    def t_urgency_on_time():
        npt.assert_allclose(compute_urgency(3600, 3600), 1.0)
    run_test("urgency: exactly on time = 1.0", t_urgency_on_time)

    def t_urgency_overdue():
        u2 = compute_urgency(7200, 3600)
        u4 = compute_urgency(14400, 3600)
        assert u2 > 1.0 and u4 > u2
    run_test("urgency: grows quadratically when overdue", t_urgency_overdue)

    def t_urgency_quadratic():
        # 2x overdue → urgency = 1 + (2-1)² = 2
        npt.assert_allclose(compute_urgency(7200, 3600), 2.0, atol=1e-10)
        # 3x overdue → urgency = 1 + (3-1)² = 5
        npt.assert_allclose(compute_urgency(10800, 3600), 5.0, atol=1e-10)
    run_test("urgency: quadratic formula correct", t_urgency_quadratic)

    def t_urgency_zero_interval():
        assert compute_urgency(100, 0) == 1.0
    run_test("urgency: zero revisit interval → 1.0", t_urgency_zero_interval)

    def t_score_priority():
        s1 = compute_score(1, 1.0, np.deg2rad(45))
        s5 = compute_score(5, 1.0, np.deg2rad(45))
        s10 = compute_score(10, 1.0, np.deg2rad(45))
        assert s1 > s5 > s10
    run_test("score: priority ordering 1 > 5 > 10", t_score_priority)

    def t_score_elevation():
        assert compute_score(5, 1.0, np.deg2rad(75)) > compute_score(5, 1.0, np.deg2rad(15))
    run_test("score: higher elevation → higher score", t_score_elevation)

    def t_score_urgency_boost():
        assert compute_score(5, 5.0, np.deg2rad(45)) > compute_score(5, 1.0, np.deg2rad(45))
    run_test("score: urgency multiplies", t_score_urgency_boost)

    def t_score_positive():
        for pri in range(1, 11):
            for el in [0.1, 0.5, 1.5]:
                assert compute_score(pri, 1.0, el) > 0
    run_test("score: always positive", t_score_positive)

    def t_vis_windows():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        wins = compute_visibility_windows(SENSOR_RADAR, sats,
                                           t.epoch_jd, t.epoch_jd + 2/24, dt_step=60)
        assert isinstance(wins, list)
        for w in wins:
            assert w.start_jd <= w.end_jd and w.peak_el > 0 and w.min_range > 0
    run_test("visibility windows valid structure", t_vis_windows)

    def t_wins_sorted():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        wins = compute_visibility_windows(SENSOR_RADAR, sats,
                                           t.epoch_jd, t.epoch_jd + 4/24, dt_step=60)
        for i in range(1, len(wins)):
            assert wins[i].start_jd >= wins[i-1].start_jd
    run_test("visibility windows sorted by time", t_wins_sorted)

    def t_schedule_e2e():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 2/24,
                              dwell_time=10, slew_time=5,
                              dt_decision=30, dt_visibility=60)
        assert isinstance(res, ScheduleResult)
        assert isinstance(res.tasks, list)
        assert isinstance(res.windows, list)
        assert isinstance(res.unscheduled_windows, list)
    run_test("schedule_greedy returns ScheduleResult", t_schedule_e2e)

    def t_schedule_stats():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 2/24,
                              dt_decision=30, dt_visibility=60)
        for k in ["total_tasks", "total_windows", "unscheduled_windows",
                   "unique_satellites_observed", "total_satellites",
                   "coverage_fraction", "schedule_duration_hours",
                   "sensor_utilization", "mean_score", "mean_elevation_deg",
                   "observation_counts", "revisit_compliance_fraction"]:
            assert k in res.stats, f"missing stat: {k}"
    run_test("schedule stats all keys present", t_schedule_stats)

    def t_schedule_task_fields():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=2, revisit_interval=1800)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 3/24,
                              dt_decision=30, dt_visibility=60)
        for task in res.tasks:
            assert task.norad_id == 25544
            assert task.dwell_time > 0
            assert task.score > 0
            assert task.elevation > 0
            assert task.slant_range > 0
            assert task.urgency >= 1.0
            assert task.start_jd < task.end_jd
    run_test("scheduled task field validity", t_schedule_task_fields)

    def t_schedule_no_overlap():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=1800)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 4/24,
                              dwell_time=15, slew_time=5,
                              dt_decision=30, dt_visibility=60)
        for i in range(1, len(res.tasks)):
            assert res.tasks[i].start_jd >= res.tasks[i-1].end_jd
    run_test("scheduled tasks don't overlap", t_schedule_no_overlap)

    def t_schedule_multi_sat():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = parse_tle("CSS", ISS_L1.replace("25544", "99999"),
                        ISS_L2.replace("210.0000", "100.0000"))
        sats = [
            SatelliteTask(tle=t1, base_priority=1, revisit_interval=1800),
            SatelliteTask(tle=t2, base_priority=5, revisit_interval=7200),
        ]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t1.epoch_jd, t1.epoch_jd + 4/24,
                              dt_decision=30, dt_visibility=60)
        assert res.stats["total_satellites"] == 2
    run_test("schedule multi-satellite catalog", t_schedule_multi_sat)

    def t_schedule_priority_dominance():
        # High-priority satellite should get more tasks
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = parse_tle("LOW", ISS_L1.replace("25544", "88888"),
                        ISS_L2.replace("210.0000", "210.1000"))  # nearly same orbit
        sats = [
            SatelliteTask(tle=t1, base_priority=1, revisit_interval=600),
            SatelliteTask(tle=t2, base_priority=10, revisit_interval=600),
        ]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t1.epoch_jd, t1.epoch_jd + 4/24,
                              dt_decision=15, dt_visibility=60)
        counts = res.stats["observation_counts"]
        if 25544 in counts and 88888 in counts:
            assert counts[25544] >= counts[88888]
    run_test("higher priority → more observations", t_schedule_priority_dominance)

    def t_format():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=3, revisit_interval=1800)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 1/24,
                              dt_decision=60, dt_visibility=60)
        report = format_schedule(res, SENSOR_RADAR)
        assert "TASK SCHEDULE" in report and "TestRadar" in report and "RADAR" in report
    run_test("format_schedule output", t_format)

    def t_precomputed_windows():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        jd_s, jd_e = t.epoch_jd, t.epoch_jd + 2/24
        wins = compute_visibility_windows(SENSOR_RADAR, sats, jd_s, jd_e, dt_step=60)
        res = schedule_greedy(SENSOR_RADAR, sats, jd_s, jd_e, dt_decision=30, windows=wins)
        assert res.stats["total_windows"] == len(wins)
    run_test("schedule with pre-computed windows", t_precomputed_windows)

    def t_empty_schedule():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        # Very short window → likely no passes
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 1/86400,  # 1 second
                              dt_decision=30, dt_visibility=30)
        assert isinstance(res.tasks, list)
    run_test("scheduler handles empty schedule gracefully", t_empty_schedule)

    def t_satellite_task_defaults():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        st = SatelliteTask(tle=t)
        assert st.base_priority == 5
        assert st.revisit_interval == 3600.0
        assert st.rcs == 1.0
    run_test("SatelliteTask defaults", t_satellite_task_defaults)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  orbital_frames — Comprehensive Unit Test Suite")
    print("=" * 60)

    test_utils()
    test_frames()
    test_orbits()
    test_coverage()
    test_sun()
    test_moon()
    test_exclusion()
    test_tle()
    test_sensor()
    test_scheduler()

    total = _results["pass"] + _results["fail"]
    print("\n" + "=" * 60)
    print(f"  TOTAL:  {_results['pass']} passed,  {_results['fail']} failed  "
          f"({total} tests)")
    if _results["errors"]:
        print(f"  FAILED: {', '.join(_results['errors'])}")
    print("=" * 60)

    sys.exit(0 if _results["fail"] == 0 else 1)


# ═══════════════════════════════════════════════════════════════════════════
#  9. SENSOR MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_sensor():
    print("\n── sensor ──")

    def t_sensor_ecef():
        r = SENSOR_RADAR.ecef()
        assert np.linalg.norm(r) > R_EARTH * 0.99
        assert len(r) == 3
    run_test("sensor ECEF position valid", t_sensor_ecef)

    def t_sensor_eci():
        r = SENSOR_RADAR.eci(JD_TEST)
        npt.assert_allclose(np.linalg.norm(r), np.linalg.norm(SENSOR_RADAR.ecef()), rtol=1e-10)
    run_test("sensor ECI magnitude matches ECEF", t_sensor_eci)

    def t_sensor_velocity():
        v = SENSOR_RADAR.eci_velocity(JD_TEST)
        assert 200 < np.linalg.norm(v) < 500
    run_test("sensor ECI velocity ~300-500 m/s", t_sensor_velocity)

    def t_sensor_equator_vel():
        s = GroundSensor(lat=0.0, lon=0.0)
        v = s.eci_velocity(JD_TEST)
        assert 400 < np.linalg.norm(v) < 500
    run_test("equatorial sensor velocity ~465 m/s", t_sensor_equator_vel)

    def t_azel_zenith():
        r_ecef = lla_to_ecef(0.0, 0.0, 0.0)
        r_target = lla_to_ecef(0.0, 0.0, 500e3)
        az, el, rng = topocentric_azel(r_ecef, 0.0, 0.0, r_target)
        npt.assert_allclose(el, np.pi / 2, atol=0.01)
        npt.assert_allclose(rng, 500e3, rtol=0.01)
    run_test("topocentric azel: zenith pass", t_azel_zenith)

    def t_azel_directional():
        # Verify azimuth convention: target due east at same lat
        lat_s, lon_s = np.deg2rad(40), np.deg2rad(0)
        r_sensor = lla_to_ecef(lat_s, lon_s, 0)
        r_east = lla_to_ecef(lat_s, np.deg2rad(5), 400e3)
        az_e, el_e, _ = topocentric_azel(r_sensor, lat_s, lon_s, r_east)
        # East target: verify az is consistent and el > 0
        assert el_e > 0
        # Verify different directions give different azimuths
        r_west = lla_to_ecef(lat_s, np.deg2rad(-5), 400e3)
        az_w, _, _ = topocentric_azel(r_sensor, lat_s, lon_s, r_west)
        assert abs(az_e - az_w) > np.deg2rad(90), "east/west should differ by >90°"
    run_test("topocentric azel: directional consistency", t_azel_directional)

    def t_azel_east():
        r_ecef = lla_to_ecef(0.0, 0.0, 0.0)
        r_target = lla_to_ecef(0.0, np.deg2rad(5), 500e3)
        az, el, rng = topocentric_azel(r_ecef, 0.0, 0.0, r_target)
        npt.assert_allclose(az, np.pi / 2, atol=np.deg2rad(15))
    run_test("topocentric azel: east azimuth", t_azel_east)

    def t_sun_elevation():
        el = sun_elevation_at_sensor(SENSOR_RADAR, JD_TEST)
        assert -np.pi / 2 <= el <= np.pi / 2
    run_test("sun elevation at sensor in bounds", t_sun_elevation)

    def t_vmag_finite():
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 1000e3), JD_TEST)
        r_sensor = GroundSensor(lat=0.0, lon=0.0).eci(JD_TEST)
        m = estimate_visual_magnitude(r_sat, r_sensor, JD_TEST, rcs=1.0)
        assert np.isfinite(m) and -5 < m < 30
    run_test("visual magnitude finite and reasonable", t_vmag_finite)

    def t_vmag_closer_brighter():
        r_s = GroundSensor(lat=0.0, lon=0.0).eci(JD_TEST)
        r1 = ecef_to_eci(lla_to_ecef(0.0, 0.0, 500e3), JD_TEST)
        r2 = ecef_to_eci(lla_to_ecef(0.0, 0.0, 2000e3), JD_TEST)
        m1 = estimate_visual_magnitude(r1, r_s, JD_TEST, rcs=1.0)
        m2 = estimate_visual_magnitude(r2, r_s, JD_TEST, rcs=1.0)
        assert m1 < m2
    run_test("closer satellite → brighter (lower mag)", t_vmag_closer_brighter)

    def t_vmag_larger_brighter():
        r_s = GroundSensor(lat=0.0, lon=0.0).eci(JD_TEST)
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 1000e3), JD_TEST)
        assert estimate_visual_magnitude(r_sat, r_s, JD_TEST, rcs=10.0) < \
               estimate_visual_magnitude(r_sat, r_s, JD_TEST, rcs=0.1)
    run_test("larger RCS → brighter", t_vmag_larger_brighter)

    def t_vis_radar_keys():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r_sat = propagate_tle(t, JD_TEST)[0]
        vis = check_visibility(SENSOR_RADAR, r_sat, JD_TEST)
        for k in ["visible", "el", "az", "range", "in_elevation", "in_azimuth",
                   "in_range", "in_for", "reasons"]:
            assert k in vis, f"missing key: {k}"
    run_test("radar visibility: all output keys", t_vis_radar_keys)

    def t_vis_optical_keys():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        r_sat = propagate_tle(t, JD_TEST)[0]
        vis = check_visibility(SENSOR_OPTICAL, r_sat, JD_TEST, rcs=400.0)
        for k in ["sky_dark", "sat_illuminated", "sun_clear", "moon_clear",
                   "visual_mag", "mag_detectable"]:
            assert k in vis, f"missing key: {k}"
    run_test("optical visibility: all output keys", t_vis_optical_keys)

    def t_vis_below_horizon():
        r_sat = -normalize(SENSOR_RADAR.eci(JD_TEST)) * (R_EARTH + 400e3)
        vis = check_visibility(SENSOR_RADAR, r_sat, JD_TEST)
        assert not vis["visible"] and "elevation" in vis["reasons"]
    run_test("satellite below horizon → not visible", t_vis_below_horizon)

    def t_vis_out_of_range():
        s = GroundSensor(name="Short", lat=0.0, lon=0.0, sensor_type="radar",
                         max_range=100e3, min_elevation=np.deg2rad(-90))
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 500e3), JD_TEST)
        vis = check_visibility(s, r_sat, JD_TEST)
        assert not vis["visible"] and "range" in vis["reasons"]
    run_test("satellite out of range → not visible", t_vis_out_of_range)

    def t_sensor_for():
        s = GroundSensor(lat=0.0, lon=0.0, sensor_type="radar",
                         field_of_regard=np.deg2rad(30), min_elevation=np.deg2rad(0))
        # Zenith target should be in FOR (zenith angle = 0 < 30°)
        r_sat = ecef_to_eci(lla_to_ecef(0.0, 0.0, 500e3), JD_TEST)
        vis = check_visibility(s, r_sat, JD_TEST)
        assert vis["in_for"]
    run_test("field of regard: zenith within 30° FOR", t_sensor_for)


# ═══════════════════════════════════════════════════════════════════════════
#  10. SCHEDULER MODULE
# ═══════════════════════════════════════════════════════════════════════════

def test_scheduler():
    print("\n── scheduler ──")

    def t_urgency_fresh():
        npt.assert_allclose(compute_urgency(0, 3600), 1.0)
    run_test("urgency: fresh = 1.0", t_urgency_fresh)

    def t_urgency_on_time():
        npt.assert_allclose(compute_urgency(3600, 3600), 1.0)
    run_test("urgency: exactly on time = 1.0", t_urgency_on_time)

    def t_urgency_overdue():
        u2 = compute_urgency(7200, 3600)
        u4 = compute_urgency(14400, 3600)
        assert u2 > 1.0 and u4 > u2
    run_test("urgency: grows quadratically when overdue", t_urgency_overdue)

    def t_urgency_quadratic():
        # 2x overdue → urgency = 1 + (2-1)² = 2
        npt.assert_allclose(compute_urgency(7200, 3600), 2.0, atol=1e-10)
        # 3x overdue → urgency = 1 + (3-1)² = 5
        npt.assert_allclose(compute_urgency(10800, 3600), 5.0, atol=1e-10)
    run_test("urgency: quadratic formula correct", t_urgency_quadratic)

    def t_urgency_zero_interval():
        assert compute_urgency(100, 0) == 1.0
    run_test("urgency: zero revisit interval → 1.0", t_urgency_zero_interval)

    def t_score_priority():
        s1 = compute_score(1, 1.0, np.deg2rad(45))
        s5 = compute_score(5, 1.0, np.deg2rad(45))
        s10 = compute_score(10, 1.0, np.deg2rad(45))
        assert s1 > s5 > s10
    run_test("score: priority ordering 1 > 5 > 10", t_score_priority)

    def t_score_elevation():
        assert compute_score(5, 1.0, np.deg2rad(75)) > compute_score(5, 1.0, np.deg2rad(15))
    run_test("score: higher elevation → higher score", t_score_elevation)

    def t_score_urgency_boost():
        assert compute_score(5, 5.0, np.deg2rad(45)) > compute_score(5, 1.0, np.deg2rad(45))
    run_test("score: urgency multiplies", t_score_urgency_boost)

    def t_score_positive():
        for pri in range(1, 11):
            for el in [0.1, 0.5, 1.5]:
                assert compute_score(pri, 1.0, el) > 0
    run_test("score: always positive", t_score_positive)

    def t_vis_windows():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        wins = compute_visibility_windows(SENSOR_RADAR, sats,
                                           t.epoch_jd, t.epoch_jd + 2/24, dt_step=60)
        assert isinstance(wins, list)
        for w in wins:
            assert w.start_jd <= w.end_jd and w.peak_el > 0 and w.min_range > 0
    run_test("visibility windows valid structure", t_vis_windows)

    def t_wins_sorted():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        wins = compute_visibility_windows(SENSOR_RADAR, sats,
                                           t.epoch_jd, t.epoch_jd + 4/24, dt_step=60)
        for i in range(1, len(wins)):
            assert wins[i].start_jd >= wins[i-1].start_jd
    run_test("visibility windows sorted by time", t_wins_sorted)

    def t_schedule_e2e():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 2/24,
                              dwell_time=10, slew_time=5,
                              dt_decision=30, dt_visibility=60)
        assert isinstance(res, ScheduleResult)
        assert isinstance(res.tasks, list)
        assert isinstance(res.windows, list)
        assert isinstance(res.unscheduled_windows, list)
    run_test("schedule_greedy returns ScheduleResult", t_schedule_e2e)

    def t_schedule_stats():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 2/24,
                              dt_decision=30, dt_visibility=60)
        for k in ["total_tasks", "total_windows", "unscheduled_windows",
                   "unique_satellites_observed", "total_satellites",
                   "coverage_fraction", "schedule_duration_hours",
                   "sensor_utilization", "mean_score", "mean_elevation_deg",
                   "observation_counts", "revisit_compliance_fraction"]:
            assert k in res.stats, f"missing stat: {k}"
    run_test("schedule stats all keys present", t_schedule_stats)

    def t_schedule_task_fields():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=2, revisit_interval=1800)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 3/24,
                              dt_decision=30, dt_visibility=60)
        for task in res.tasks:
            assert task.norad_id == 25544
            assert task.dwell_time > 0
            assert task.score > 0
            assert task.elevation > 0
            assert task.slant_range > 0
            assert task.urgency >= 1.0
            assert task.start_jd < task.end_jd
    run_test("scheduled task field validity", t_schedule_task_fields)

    def t_schedule_no_overlap():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=1800)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 4/24,
                              dwell_time=15, slew_time=5,
                              dt_decision=30, dt_visibility=60)
        for i in range(1, len(res.tasks)):
            assert res.tasks[i].start_jd >= res.tasks[i-1].end_jd
    run_test("scheduled tasks don't overlap", t_schedule_no_overlap)

    def t_schedule_multi_sat():
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = parse_tle("CSS", ISS_L1.replace("25544", "99999"),
                        ISS_L2.replace("210.0000", "100.0000"))
        sats = [
            SatelliteTask(tle=t1, base_priority=1, revisit_interval=1800),
            SatelliteTask(tle=t2, base_priority=5, revisit_interval=7200),
        ]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t1.epoch_jd, t1.epoch_jd + 4/24,
                              dt_decision=30, dt_visibility=60)
        assert res.stats["total_satellites"] == 2
    run_test("schedule multi-satellite catalog", t_schedule_multi_sat)

    def t_schedule_priority_dominance():
        # High-priority satellite should get more tasks
        t1 = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        t2 = parse_tle("LOW", ISS_L1.replace("25544", "88888"),
                        ISS_L2.replace("210.0000", "210.1000"))  # nearly same orbit
        sats = [
            SatelliteTask(tle=t1, base_priority=1, revisit_interval=600),
            SatelliteTask(tle=t2, base_priority=10, revisit_interval=600),
        ]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t1.epoch_jd, t1.epoch_jd + 4/24,
                              dt_decision=15, dt_visibility=60)
        counts = res.stats["observation_counts"]
        if 25544 in counts and 88888 in counts:
            assert counts[25544] >= counts[88888]
    run_test("higher priority → more observations", t_schedule_priority_dominance)

    def t_format():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=3, revisit_interval=1800)]
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 1/24,
                              dt_decision=60, dt_visibility=60)
        report = format_schedule(res, SENSOR_RADAR)
        assert "TASK SCHEDULE" in report and "TestRadar" in report and "RADAR" in report
    run_test("format_schedule output", t_format)

    def t_precomputed_windows():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        jd_s, jd_e = t.epoch_jd, t.epoch_jd + 2/24
        wins = compute_visibility_windows(SENSOR_RADAR, sats, jd_s, jd_e, dt_step=60)
        res = schedule_greedy(SENSOR_RADAR, sats, jd_s, jd_e, dt_decision=30, windows=wins)
        assert res.stats["total_windows"] == len(wins)
    run_test("schedule with pre-computed windows", t_precomputed_windows)

    def t_empty_schedule():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        sats = [SatelliteTask(tle=t, base_priority=1, revisit_interval=3600)]
        # Very short window → likely no passes
        res = schedule_greedy(SENSOR_RADAR, sats,
                              t.epoch_jd, t.epoch_jd + 1/86400,  # 1 second
                              dt_decision=30, dt_visibility=30)
        assert isinstance(res.tasks, list)
    run_test("scheduler handles empty schedule gracefully", t_empty_schedule)

    def t_satellite_task_defaults():
        t = parse_tle(ISS_NAME, ISS_L1, ISS_L2)
        st = SatelliteTask(tle=t)
        assert st.base_priority == 5
        assert st.revisit_interval == 3600.0
        assert st.rcs == 1.0
    run_test("SatelliteTask defaults", t_satellite_task_defaults)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  astrosor — Comprehensive Unit Test Suite")
    print("=" * 60)

    test_utils()
    test_frames()
    test_orbits()
    test_coverage()
    test_sun()
    test_moon()
    test_exclusion()
    test_tle()
    test_sensor()
    test_scheduler()

    total = _results["pass"] + _results["fail"]
    print("\n" + "=" * 60)
    print(f"  TOTAL:  {_results['pass']} passed,  {_results['fail']} failed  "
          f"({total} tests)")
    if _results["errors"]:
        print(f"  FAILED: {', '.join(_results['errors'])}")
    print("=" * 60)

    sys.exit(0 if _results["fail"] == 0 else 1)
