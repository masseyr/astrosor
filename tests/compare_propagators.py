#!/usr/bin/env python
"""
compare_propagators.py — Propagator accuracy comparison
========================================================

Compares three propagation methods side-by-side:

  1. J2 + drag   (astrosor — secular J2 + linear drag via ndot)
  2. Two-body    (astrosor.orbits.propagate_kepler — pure Keplerian)
  3. SGP4        (sgp4 library + astropy TEME→ITRS frame conversion)

SGP4 is taken as the reference because it is the standard and its TLE
elements are defined specifically for that propagator.  The generated
TLE (from from_lat_lon_alt or tle_to_lines) is fed into both astrosor
and sgp4, so the three methods share the same epoch elements.

Dependencies
------------
    pip install sgp4 astropy matplotlib   # optional but needed for full output

Run
---
    python tests/compare_propagators.py
"""

from __future__ import annotations
import numpy as np

# ── astrosor ──────────────────────────────────────────────────────────────────
from astrosor import julian_date
from astrosor.satellite import Satellite
from astrosor.tle import tle_epoch_state, tle_to_lines
from astrosor.orbits import propagate_kepler
from astrosor.utils import eci_to_ecef

# ── SGP4 + astropy (optional) ─────────────────────────────────────────────────
try:
    from sgp4.api import Satrec
    from astropy.time import Time
    from astropy.coordinates import (
        TEME, ITRS, CartesianRepresentation, CartesianDifferential,
    )
    import astropy.units as u
    HAS_SGP4 = True
except ImportError:
    HAS_SGP4 = False
    print("NOTE: sgp4 and/or astropy not installed — SGP4 column will be absent.")
    print("      pip install sgp4 astropy\n")

# ── matplotlib (optional) ─────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sgp4_ecef(satrec, jd: float):
    """SGP4 propagation → ECEF position [m] and velocity [m/s].

    Converts from TEME (SGP4 native) to ITRS via astropy, which applies the
    proper nutation/polar motion transforms rather than a simple GMST rotation.
    """
    t = Time(jd, format="jd", scale="utc")
    err, r_km, v_km_s = satrec.sgp4(t.jd1, t.jd2)
    if err != 0:
        return None, None
    r_teme = CartesianRepresentation(r_km * u.km)
    v_teme = CartesianDifferential(v_km_s * (u.km / u.s))
    itrs = TEME(r_teme.with_differentials(v_teme), obstime=t).transform_to(
        ITRS(obstime=t)
    )
    r_ecef = itrs.cartesian.xyz.to(u.m).value
    v_ecef = itrs.velocity.d_xyz.to(u.m / u.s).value
    return r_ecef, v_ecef


def _fmt(d_km: np.ndarray, n: int) -> str:
    """One-line summary: values at 6h, 12h, 24h + RMS + max."""
    i6  = n //  4
    i12 = n //  2
    i24 = n  - 1
    rms = np.sqrt(np.mean(d_km ** 2))
    return (
        f"6h={d_km[i6]:7.3f}  12h={d_km[i12]:7.3f}  "
        f"24h={d_km[i24]:7.3f}  rms={rms:7.3f}  max={d_km.max():7.3f}  [km]"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Core comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    sat: Satellite,
    label: str,
    hours: float = 24.0,
    n: int = 145,
) -> dict:
    """Propagate with all three methods and report position differences.

    Parameters
    ----------
    sat   : Satellite — source object (TLE epoch is used as t=0)
    label : str       — display name for this run
    hours : float     — propagation window [h]
    n     : int       — number of time steps

    Returns
    -------
    dict with keys 'j2_vs_2b', 'j2_vs_sgp4', '2b_vs_sgp4'  (each (n,) km array)
    """
    epoch  = sat.epoch_jd
    dt_s   = np.linspace(0.0, hours * 3600.0, n)
    jds    = epoch + dt_s / 86400.0

    # Epoch state for two-body (J2 epoch state as starting point)
    r0_eci, v0_eci = tle_epoch_state(sat.tle)

    # SGP4 — built from the same TLE elements via tle_to_lines so all
    # three methods share identical epoch elements
    if HAS_SGP4:
        l1, l2 = tle_to_lines(sat.tle)
        satrec  = Satrec.twoline2rv(l1, l2)

    r_j2   = np.empty((n, 3))
    r_2b   = np.empty((n, 3))
    r_sgp4 = np.empty((n, 3)) if HAS_SGP4 else None

    for i, (jd, dt) in enumerate(zip(jds, dt_s)):
        # 1) J2 + drag
        r_eci_j2, _ = sat.state_eci(jd)
        r_j2[i]     = eci_to_ecef(r_eci_j2, jd)

        # 2) Two-body Keplerian (propagated from epoch ECI state)
        r_eci_2b, _ = propagate_kepler(r0_eci, v0_eci, dt)
        r_2b[i]     = eci_to_ecef(r_eci_2b, jd)

        # 3) SGP4 via astropy TEME→ITRS
        if HAS_SGP4:
            r_ecef_sgp4, _ = _sgp4_ecef(satrec, jd)
            if r_ecef_sgp4 is not None:
                r_sgp4[i] = r_ecef_sgp4

    # ── Distance arrays [km] ─────────────────────────────────────────────────
    d_j2_2b  = np.linalg.norm(r_j2 - r_2b,   axis=1) / 1e3
    d_j2_s4  = np.linalg.norm(r_j2 - r_sgp4, axis=1) / 1e3 if HAS_SGP4 else None
    d_2b_s4  = np.linalg.norm(r_2b - r_sgp4, axis=1) / 1e3 if HAS_SGP4 else None

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  {label}")
    print(
        f"  a={sat.semi_major_axis/1e3:.1f} km  e={sat.eccentricity:.5f}  "
        f"i={np.rad2deg(sat.inclination):.2f}°  "
        f"T={sat.period:.1f} min  {sat.orbit_type}"
    )
    print(f"{'─'*72}")
    print(f"  {'Method pair':<28}  {'6h':>7}  {'12h':>7}  {'24h':>7}  "
          f"{'RMS':>7}  {'Max':>7}  unit")
    print(f"{'─'*72}")
    print(f"  {'J2+drag  vs  Two-body':<28}  {_fmt(d_j2_2b, n)}")
    if HAS_SGP4:
        print(f"  {'J2+drag  vs  SGP4':<28}  {_fmt(d_j2_s4, n)}")
        print(f"  {'Two-body vs  SGP4':<28}  {_fmt(d_2b_s4, n)}")
    print(f"{'─'*72}")

    # ── Note on reference frame difference ───────────────────────────────────
    print(
        "  Note: astrosor uses a GMST-based ECI→ECEF rotation; SGP4 output is\n"
        "  TEME, converted to ITRS via astropy (includes nutation/polar motion).\n"
        "  A ~10–100 m systematic offset at epoch is expected from this alone."
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    if HAS_MPL:
        hrs = dt_s / 3600.0
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.semilogy(hrs, d_j2_2b, label="J2+drag vs Two-body", lw=1.5)
        if HAS_SGP4:
            ax.semilogy(hrs, d_j2_s4, label="J2+drag vs SGP4", lw=1.5)
            ax.semilogy(hrs, d_2b_s4, label="Two-body vs SGP4", lw=1.5, ls="--")
        ax.set(
            xlabel="Time [h]",
            ylabel="Position error [km]  (log)",
            title=f"Propagator comparison — {label}",
        )
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        fname = f"prop_cmp_{label.split()[0]}.png"
        plt.savefig(fname, dpi=120)
        print(f"  Plot saved → {fname}")
        plt.close()

    return {"j2_vs_2b": d_j2_2b, "j2_vs_sgp4": d_j2_s4, "2b_vs_sgp4": d_2b_s4}


# ─────────────────────────────────────────────────────────────────────────────
#  Round-trip: from_lat_lon_alt → lat_lon_alt at the same epoch
# ─────────────────────────────────────────────────────────────────────────────

def test_from_lat_lon_alt_round_trip(jd: float):
    """Verify that from_lat_lon_alt recovers the input position at epoch."""
    print(f"\n{'═'*72}")
    print("  Round-trip: from_lat_lon_alt → lat_lon_alt at epoch")
    print(f"{'─'*72}")
    print(f"  {'Case':<28}  {'Δlat [m]':>9}  {'Δlon [m]':>9}  {'Δalt [m]':>9}  OK?")
    print(f"{'─'*72}")

    cases = [
        dict(label="ISS-like LEO",      lat=48.85,  lon=  2.35, alt=  418_000, inc=51.6,  mm=15.50, asc=True),
        dict(label="ISS-like (desc)",   lat=48.85,  lon=  2.35, alt=  418_000, inc=51.6,  mm=15.50, asc=False),
        dict(label="Sun-sync LEO",      lat=70.00,  lon= 45.00, alt=  700_000, inc=98.2,  mm=14.22, asc=True),
        dict(label="Equatorial",        lat= 2.00,  lon=100.00, alt=  550_000, inc=10.0,  mm=15.17, asc=True),
        dict(label="GPS-like MEO",      lat=20.00,  lon= 60.00, alt=20_200_000,inc=55.0,  mm= 2.006,asc=True),
    ]

    all_pass = True
    for c in cases:
        try:
            sat = Satellite.from_lat_lon_alt(
                lat=c["lat"], lon=c["lon"], alt=c["alt"], jd=jd,
                inclination=c["inc"], mean_motion=c["mm"],
                ascending=c["asc"], name=c["label"],
            )
            lat_o, lon_o, alt_o = sat.lat_lon_alt(jd)

            deg_km   = 111.32
            dlat_m   = abs(c["lat"] - lat_o) * deg_km * 1e3
            dlon_m   = abs(c["lon"] - lon_o) * deg_km * np.cos(np.deg2rad(c["lat"])) * 1e3
            dalt_m   = abs(c["alt"] - alt_o)
            ok       = max(dlat_m, dlon_m) < 100.0 and dalt_m < 50.0
            all_pass = all_pass and ok
            flag     = "✓" if ok else "✗"
            print(
                f"  {flag} {c['label']:<26}  "
                f"{dlat_m:9.2f}  {dlon_m:9.2f}  {dalt_m:9.2f}"
            )
        except Exception as exc:
            all_pass = False
            print(f"  ✗ {c['label']:<26}  ERROR: {exc}")

    print(f"{'─'*72}")
    print(f"  {'All passed' if all_pass else 'Some FAILED'}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  Sensitivity: how does J2 error grow vs orbital altitude?
# ─────────────────────────────────────────────────────────────────────────────

def altitude_sensitivity(jd: float):
    """Show 24h J2-vs-SGP4 RMS error across altitude bands."""
    if not HAS_SGP4:
        return

    print(f"\n{'═'*72}")
    print("  24-h J2+drag vs SGP4 RMS error by altitude")
    print(f"{'─'*72}")
    print(f"  {'Orbit':<22}  {'alt [km]':>9}  {'J2 RMS [km]':>12}  {'2B RMS [km]':>12}")
    print(f"{'─'*72}")

    orbits = [
        dict(label="LEO 300 km",    alt=  300_000, inc=51.6,  mm=15.95),
        dict(label="LEO 500 km",    alt=  500_000, inc=51.6,  mm=15.59),
        dict(label="ISS-like",      alt=  418_000, inc=51.6,  mm=15.50),
        dict(label="Sun-sync 700",  alt=  700_000, inc=98.2,  mm=14.22),
        dict(label="MEO 10000 km",  alt=10_000_000,inc=55.0,  mm= 5.37),
        dict(label="GPS 20200 km",  alt=20_200_000,inc=55.0,  mm= 2.006),
    ]

    for o in orbits:
        try:
            sat = Satellite.from_lat_lon_alt(
                lat=30.0, lon=0.0, alt=o["alt"], jd=jd,
                inclination=o["inc"], mean_motion=o["mm"],
                bstar=4e-5 if o["alt"] < 1_000_000 else 0.0,
                ndot= 2e-4 if o["alt"] < 1_000_000 else 0.0,
                ascending=True,
            )
            res   = compare(sat, o["label"], hours=24, n=97)
            rms_j2 = np.sqrt(np.mean(res["j2_vs_sgp4"] ** 2))
            rms_2b = np.sqrt(np.mean(res["2b_vs_sgp4"] ** 2))
            print(
                f"  {o['label']:<22}  {o['alt']/1e3:>9.0f}  "
                f"{rms_j2:>12.3f}  {rms_2b:>12.3f}"
            )
        except Exception as exc:
            print(f"  {o['label']:<22}  ERROR: {exc}")
    print(f"{'─'*72}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    JD = julian_date(2026, 2, 24, 12, 0, 0)

    # ── Three representative orbits, full comparison ──────────────────────────
    orbits = [
        dict(
            label="ISS-like LEO",
            lat=40.0, lon=10.0, alt=418_000,
            inclination=51.6, mean_motion=15.50,
            bstar=4.5e-5, ndot=2.2e-4, ascending=True,
        ),
        dict(
            label="Sun-sync LEO",
            lat=55.0, lon=30.0, alt=700_000,
            inclination=98.2, mean_motion=14.22,
            bstar=1.0e-5, ndot=5.0e-5, ascending=True,
        ),
        dict(
            label="GPS-like MEO",
            lat=20.0, lon=60.0, alt=20_200_000,
            inclination=55.0, mean_motion=2.0057,
            ascending=True,
        ),
    ]

    for kw in orbits:
        label = kw.pop("label")
        sat   = Satellite.from_lat_lon_alt(jd=JD, name=label, **kw)
        compare(sat, label, hours=24, n=145)

    # ── Round-trip accuracy ───────────────────────────────────────────────────
    test_from_lat_lon_alt_round_trip(JD)

    # ── Altitude sensitivity (only with SGP4) ─────────────────────────────────
    altitude_sensitivity(JD)
