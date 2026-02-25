"""
example_coverage_analysis.py — Demonstration of orbital_frames Library
=======================================================================

Demonstrates UNW/TNW coordinate transformations and sensor coverage
analysis for a nadir-pointing sensor on a LEO satellite.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astrosor import (
    # Frame transforms
    eci_to_unw_matrix, eci_to_tnw_matrix,
    eci_to_unw, eci_to_tnw, unw_to_tnw,
    # Covariance
    transform_covariance_eci_to_unw, transform_covariance_eci_to_tnw,
    # Orbits
    keplerian_to_eci, eci_to_keplerian, propagate_kepler,
    compute_orbital_period,
    # Coverage
    sensor_footprint_unw, sensor_footprint_tnw,
    ground_trace, access_geometry, coverage_windows,
    earth_intersection,
    # Utils
    normalize, R_EARTH, MU_EARTH,
    eci_to_ecef, ecef_to_lla, lla_to_ecef, ecef_to_eci,
    julian_date,
)
from astrosor.orbits import propagate_j2


def main():
    print("=" * 70)
    print("  orbital_frames — UNW/TNW Coordinate System Library Demo")
    print("=" * 70)

    # ── 1. Define a Sensor Satellite Orbit ──────────────────────────────
    print("\n1. ORBIT DEFINITION")
    print("-" * 40)

    alt_km = 500.0          # 500 km LEO
    a = R_EARTH + alt_km * 1e3
    e = 0.001               # near-circular
    inc = np.deg2rad(97.4)  # sun-synchronous-ish
    raan = np.deg2rad(45.0)
    argp = np.deg2rad(0.0)
    nu = np.deg2rad(0.0)    # start at periapsis

    r_eci, v_eci = keplerian_to_eci(a, e, inc, raan, argp, nu)
    T = compute_orbital_period(a)

    print(f"  Semi-major axis:  {a/1e3:.1f} km")
    print(f"  Altitude:         {alt_km:.0f} km")
    print(f"  Inclination:      {np.rad2deg(inc):.1f}°")
    print(f"  Period:           {T/60:.1f} min")
    print(f"  Position (ECI):   [{r_eci[0]/1e3:.1f}, {r_eci[1]/1e3:.1f}, {r_eci[2]/1e3:.1f}] km")
    print(f"  Velocity (ECI):   [{v_eci[0]/1e3:.3f}, {v_eci[1]/1e3:.3f}, {v_eci[2]/1e3:.3f}] km/s")

    # ── 2. UNW and TNW Frame Construction ───────────────────────────────
    print("\n2. LOCAL ORBITAL FRAMES")
    print("-" * 40)

    R_unw = eci_to_unw_matrix(r_eci, v_eci)
    R_tnw = eci_to_tnw_matrix(r_eci, v_eci)

    print("  UNW DCM (ECI → UNW):")
    for i, label in enumerate(["U (radial)    ", "N (along-track)", "W (cross-track)"]):
        print(f"    {label}: [{R_unw[i,0]:+.6f}, {R_unw[i,1]:+.6f}, {R_unw[i,2]:+.6f}]")

    print("\n  TNW DCM (ECI → TNW):")
    for i, label in enumerate(["T (tangential) ", "N (normal)     ", "W (cross-track)"]):
        print(f"    {label}: [{R_tnw[i,0]:+.6f}, {R_tnw[i,1]:+.6f}, {R_tnw[i,2]:+.6f}]")

    # Verify orthogonality
    err = np.max(np.abs(R_unw @ R_unw.T - np.eye(3)))
    print(f"\n  UNW orthogonality error: {err:.2e}")
    print(f"  W-axis agreement (UNW vs TNW): {np.linalg.norm(R_unw[2]-R_tnw[2]):.2e}")

    # ── 3. Vector Transformations ───────────────────────────────────────
    print("\n3. VECTOR TRANSFORMATIONS")
    print("-" * 40)

    # A relative position vector (e.g., another satellite 10 km ahead, 2 km above)
    delta_unw = np.array([2000.0, 10000.0, 0.0])  # [U, N, W] in meters
    print(f"  Relative position in UNW:  U={delta_unw[0]/1e3:.1f} km, "
          f"N={delta_unw[1]/1e3:.1f} km, W={delta_unw[2]/1e3:.1f} km")

    delta_tnw = unw_to_tnw(delta_unw, r_eci, v_eci)
    print(f"  Same vector in TNW:        T={delta_tnw[0]/1e3:.3f} km, "
          f"N={delta_tnw[1]/1e3:.3f} km, W={delta_tnw[2]/1e3:.3f} km")

    # ── 4. Covariance Transformation ────────────────────────────────────
    print("\n4. COVARIANCE TRANSFORMATION")
    print("-" * 40)

    # Typical orbit determination covariance in ECI (diagonal for illustration)
    sigma_r = 50.0   # 50 m position uncertainty
    sigma_v = 0.005  # 5 mm/s velocity uncertainty
    P_eci = np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)

    P_unw = transform_covariance_eci_to_unw(P_eci, r_eci, v_eci)
    P_tnw = transform_covariance_eci_to_tnw(P_eci, r_eci, v_eci)

    print(f"  ECI position sigmas:  [{sigma_r:.0f}, {sigma_r:.0f}, {sigma_r:.0f}] m")
    unw_sig = np.sqrt(np.diag(P_unw[:3, :3]))
    tnw_sig = np.sqrt(np.diag(P_tnw[:3, :3]))
    print(f"  UNW position sigmas:  [U={unw_sig[0]:.1f}, N={unw_sig[1]:.1f}, W={unw_sig[2]:.1f}] m")
    print(f"  TNW position sigmas:  [T={tnw_sig[0]:.1f}, N={tnw_sig[1]:.1f}, W={tnw_sig[2]:.1f}] m")
    print(f"  Trace preserved:      ECI={np.trace(P_eci[:3,:3]):.1f}, "
          f"UNW={np.trace(P_unw[:3,:3]):.1f}, TNW={np.trace(P_tnw[:3,:3]):.1f}")

    # ── 5. Sensor Footprint ─────────────────────────────────────────────
    print("\n5. SENSOR FOOTPRINT ANALYSIS")
    print("-" * 40)

    half_cone = np.deg2rad(30.0)  # 30° half-cone angle
    jd = julian_date(2026, 2, 24, 12, 0, 0)

    # Nadir-pointing sensor (boresight = -U in UNW)
    fp = sensor_footprint_unw(r_eci, v_eci, [-1, 0, 0], half_cone,
                              n_points=72, jd=jd)
    n_valid = np.sum(fp["valid"])
    print(f"  Sensor: nadir-pointing, {np.rad2deg(half_cone):.0f}° half-cone")
    print(f"  Footprint points on Earth: {n_valid}/{72}")

    if fp["center_lla"] is not None:
        c = fp["center_lla"]
        print(f"  Sub-satellite point: lat={np.rad2deg(c[0]):.2f}°, "
              f"lon={np.rad2deg(c[1]):.2f}°, alt={c[2]:.0f} m")

    if n_valid > 0:
        valid_pts = fp["lla"][fp["valid"]]
        lat_range = np.rad2deg(np.ptp(valid_pts[:, 0]))
        lon_range = np.rad2deg(np.ptp(valid_pts[:, 1]))
        print(f"  Footprint extent: Δlat≈{lat_range:.1f}°, Δlon≈{lon_range:.1f}°")

    # Forward-looking sensor (boresight = +T in TNW, tilted 20° from nadir)
    # Boresight in TNW: mix of -N (nadir-ish) and +T (forward)
    tilt = np.deg2rad(20.0)
    boresight_tnw = np.array([np.sin(tilt), -np.cos(tilt), 0.0])
    fp2 = sensor_footprint_tnw(r_eci, v_eci, boresight_tnw,
                               np.deg2rad(15.0), n_points=72, jd=jd)
    print(f"\n  Forward-tilted sensor ({np.rad2deg(tilt):.0f}° from nadir in TNW):")
    print(f"  Footprint points: {np.sum(fp2['valid'])}/{72}")

    # ── 6. Access Geometry ──────────────────────────────────────────────
    print("\n6. ACCESS GEOMETRY TO GROUND TARGET")
    print("-" * 40)

    # Target: Washington DC
    lat_dc = np.deg2rad(38.9072)
    lon_dc = np.deg2rad(-77.0369)
    r_dc_ecef = lla_to_ecef(lat_dc, lon_dc, 0.0)
    r_dc_eci = ecef_to_eci(r_dc_ecef, jd)

    geom = access_geometry(r_eci, v_eci, r_dc_eci)
    print(f"  Target: Washington DC (38.91°N, 77.04°W)")
    print(f"  Slant range:      {geom['range']/1e3:.1f} km")
    print(f"  Elevation:        {np.rad2deg(geom['elevation']):.2f}°")
    print(f"  Off-nadir angle:  {np.rad2deg(geom['off_boresight_nadir']):.2f}°")
    print(f"  LOS in UNW:       [{geom['los_unw'][0]:.4f}, "
          f"{geom['los_unw'][1]:.4f}, {geom['los_unw'][2]:.4f}]")
    print(f"  LOS in TNW:       [{geom['los_tnw'][0]:.4f}, "
          f"{geom['los_tnw'][1]:.4f}, {geom['los_tnw'][2]:.4f}]")

    # ── 7. Ground Trace ─────────────────────────────────────────────────
    print("\n7. GROUND TRACE (1 orbit)")
    print("-" * 40)

    trace = ground_trace(r_eci, v_eci, T, n_points=360, jd_epoch=jd, use_j2=True)
    print(f"  Points computed: {len(trace)}")
    print(f"  Latitude range:  {np.rad2deg(trace[:,0].min()):.1f}° to "
          f"{np.rad2deg(trace[:,0].max()):.1f}°")
    print(f"  Longitude range: {np.rad2deg(trace[:,1].min()):.1f}° to "
          f"{np.rad2deg(trace[:,1].max()):.1f}°")

    # ── 8. Coverage Windows ─────────────────────────────────────────────
    print("\n8. COVERAGE WINDOW ANALYSIS (24 hours)")
    print("-" * 40)

    duration = 86400.0  # 24 hours

    # Target function: DC stays fixed in ECEF, need to rotate to ECI at each time
    def dc_eci_at_t(t):
        jd_t = jd + t / 86400.0
        return ecef_to_eci(r_dc_ecef, jd_t)

    windows = coverage_windows(
        r_eci, v_eci, dc_eci_at_t,
        duration=duration,
        half_cone_angle=np.deg2rad(45.0),  # wide FOV
        boresight_unw=np.array([-1.0, 0.0, 0.0]),
        dt_step=30.0,
        jd_epoch=jd,
        use_j2=True,
    )

    print(f"  Target: Washington DC")
    print(f"  Sensor: nadir, 45° half-cone")
    print(f"  Coverage windows found: {len(windows)}")
    for k, w in enumerate(windows):
        print(f"    Window {k+1}: t={w['start']/60:.1f}–{w['end']/60:.1f} min, "
              f"dur={w['duration']:.0f}s, "
              f"max_el={np.rad2deg(w['max_elevation']):.1f}°, "
              f"min_range={w['min_range']/1e3:.0f} km")

    # ── 9. Multi-Sensor Comparison ──────────────────────────────────────
    print("\n9. FRAME COMPARISON: UNW vs TNW FOR ELLIPTICAL ORBIT")
    print("-" * 40)

    # Molniya orbit at true anomaly = 90° — T and N directions differ significantly
    a_mol = 26_600e3
    e_mol = 0.74
    r_mol, v_mol = keplerian_to_eci(a_mol, e_mol, np.deg2rad(63.4),
                                     np.deg2rad(0), np.deg2rad(270),
                                     np.deg2rad(90))  # at 90° true anomaly

    R_unw_mol = eci_to_unw_matrix(r_mol, v_mol)
    R_tnw_mol = eci_to_tnw_matrix(r_mol, v_mol)

    # Angle between T and N_unw (shows frame misalignment for eccentric orbits)
    T_hat = R_tnw_mol[0, :]
    N_unw_hat = R_unw_mol[1, :]
    angle_TN = np.rad2deg(np.arccos(np.clip(np.dot(T_hat, N_unw_hat), -1, 1)))
    print(f"  Molniya orbit at ν=90° (e={e_mol})")
    print(f"  Angle between T (TNW) and N (UNW): {angle_TN:.2f}°")
    print(f"  → For circular: ≈0°; for eccentric: significant deviation")
    print(f"  → This is why frame choice matters for sensor pointing!")

    print("\n" + "=" * 70)
    print("  Demo complete. All frame operations use only NumPy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
