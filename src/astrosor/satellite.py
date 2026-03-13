"""
astrosor.satellite — High-level Satellite wrapper
==================================================

Wraps a TLE with convenient properties for all Keplerian elements and
methods for state propagation and coordinate transformation.
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .tle import TLE, propagate_tle, _jd_to_epoch
from .orbits import eci_to_keplerian
from .utils import MU_EARTH, eci_to_ecef, ecef_to_eci, ecef_to_lla, lla_to_ecef, OMEGA_EARTH, gmst
from .vector import Vector3


@dataclass
class Satellite:
    """Satellite propagator and coordinate converter built on a TLE.

    Keplerian elements are epoch values taken directly from the TLE.
    All propagation methods accept a Julian Date.

    Parameters
    ----------
    tle : TLE — parsed TLE (see ``parse_tle``)
    """

    tle: TLE

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.tle.name

    @property
    def norad_id(self) -> int:
        return self.tle.norad_id

    @property
    def catalog_id(self) -> str:
        return self.tle.catalog_id

    @property
    def epoch_jd(self) -> float:
        """TLE epoch as Julian Date."""
        return self.tle.epoch_jd

    # ── Keplerian elements at epoch ───────────────────────────────────────────

    @property
    def semi_major_axis(self) -> float:
        """Semi-major axis [m]."""
        return self.tle.semi_major_axis

    @property
    def eccentricity(self) -> float:
        """Eccentricity [-]."""
        return self.tle.eccentricity

    @property
    def inclination(self) -> float:
        """Inclination [rad]."""
        return self.tle.inclination

    @property
    def raan(self) -> float:
        """Right ascension of ascending node [rad]."""
        return self.tle.raan

    @property
    def argp(self) -> float:
        """Argument of perigee [rad]."""
        return self.tle.argp

    @property
    def mean_anomaly(self) -> float:
        """Mean anomaly at epoch [rad]."""
        return self.tle.mean_anomaly

    @property
    def mean_motion(self) -> float:
        """Mean motion [rev/day]."""
        return self.tle.mean_motion

    @property
    def period(self) -> float:
        """Orbital period [min]."""
        return self.tle.period

    @property
    def apogee(self) -> float | None:
        """Apogee altitude [km] above WGS-84 ellipsoid."""
        return self.tle.apogee

    @property
    def perigee(self) -> float | None:
        """Perigee altitude [km] above WGS-84 ellipsoid."""
        return self.tle.perigee

    @property
    def orbit_type(self) -> str:
        """Orbit classification: LEO, MEO, HEO, or GEO."""
        return self.tle.orbit_type

    # ── Propagated state ──────────────────────────────────────────────────────

    def state_eci(self, jd: float) -> tuple[NDArray, NDArray]:
        """ECI position and velocity at Julian Date *jd*.

        Returns
        -------
        r_eci : (3,) ndarray — position [m]
        v_eci : (3,) ndarray — velocity [m/s]
        """
        return propagate_tle(self.tle, jd)

    def state_ecef(self, jd: float) -> tuple[NDArray, NDArray]:
        """ECEF position and velocity at Julian Date *jd*.

        Velocity accounts for Earth's rotation::

            v_ecef = R · v_eci − ω⊕ × r_ecef

        Returns
        -------
        r_ecef : (3,) ndarray — position [m]
        v_ecef : (3,) ndarray — velocity [m/s]
        """
        r_eci, v_eci = propagate_tle(self.tle, jd)
        r_ecef = eci_to_ecef(r_eci, jd)

        theta = gmst(jd)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        omega = np.array([0.0, 0.0, OMEGA_EARTH])
        v_ecef = R @ v_eci - np.cross(omega, r_ecef)

        return r_ecef, v_ecef

    def eci_position(self, jd: float) -> NDArray:
        """ECI position vector [m] at Julian Date *jd*."""
        r, _ = propagate_tle(self.tle, jd)
        return r

    def ecef_position(self, jd: float) -> NDArray:
        """ECEF position vector [m] at Julian Date *jd*."""
        r, _ = self.state_ecef(jd)
        return r

    def ecef_velocity(self, jd: float) -> NDArray:
        """ECEF velocity vector [m/s] at Julian Date *jd*.

        Accounts for Earth's rotation (Coriolis correction).
        """
        _, v = self.state_ecef(jd)
        return v

    def lat_lon_alt(self, jd: float) -> tuple[float, float, float]:
        """Geodetic position at Julian Date *jd*.

        Returns
        -------
        lat : float — geodetic latitude [deg],  positive North
        lon : float — longitude [deg],           positive East
        alt : float — altitude above WGS-84 ellipsoid [m]
        """
        r_ecef = self.ecef_position(jd)
        lla = ecef_to_lla(r_ecef)
        return float(np.rad2deg(lla[0])), float(np.rad2deg(lla[1])), float(lla[2])

    def p(self, jd: float) -> Vector3:
        """ECR position vector [m] at Julian Date *jd* as a Vector3."""
        return Vector3(self.ecef_position(jd))

    def v(self, jd: float) -> Vector3:
        """ECR velocity vector [m/s] at Julian Date *jd* as a Vector3."""
        return Vector3(self.ecef_velocity(jd))

    def keplerian_at(self, jd: float) -> dict:
        """Osculating Keplerian elements at Julian Date *jd*.

        Returns
        -------
        dict with keys: a [m], e, i [rad], raan [rad], argp [rad],
        nu [rad], E [rad], M [rad], p [m], h_mag [m²/s]
        """
        r, v = propagate_tle(self.tle, jd)
        return eci_to_keplerian(r, v)

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_tle(cls, tle: TLE) -> "Satellite":
        return cls(tle=tle)

    @classmethod
    def from_lat_lon_alt(
        cls,
        lat: float,
        lon: float,
        alt: float,
        jd: float,
        inclination: float,
        eccentricity: float = 0.0,
        mean_motion: float | None = None,
        semi_major_axis: float | None = None,
        argp: float = 0.0,
        ascending: bool = True,
        name: str = "",
        norad_id: int = 0,
        ndot: float = 0.0,
        bstar: float = 0.0,
    ) -> "Satellite":
        """Construct a Satellite whose position at *jd* matches lat/lon/alt.

        Given a geographic position and orbital shape parameters, this method
        solves for RAAN and mean anomaly so the satellite passes through that
        location at the specified epoch.

        Parameters
        ----------
        lat : float — geodetic latitude [deg], positive North
        lon : float — longitude [deg], positive East
        alt : float — altitude above WGS-84 ellipsoid [m]
        jd  : float — Julian Date of the position (becomes TLE epoch)
        inclination : float — orbital inclination [deg]
        eccentricity : float — orbital eccentricity (default 0 = circular)
        mean_motion : float | None — mean motion [rev/day]; provide this OR semi_major_axis
        semi_major_axis : float | None — semi-major axis [m]
        argp : float — argument of perigee [deg] (default 0; degenerate for circular orbits)
        ascending : bool — True if the satellite is moving northward at the given position
        name : str — satellite name
        norad_id : int — NORAD catalog number
        ndot : float — first derivative of mean motion [rev/day²]
        bstar : float — B* drag coefficient

        Returns
        -------
        Satellite

        Notes
        -----
        The given altitude must be consistent with the orbital shape; a
        UserWarning is raised if the discrepancy exceeds 1%.
        For circular orbits (e=0), *argp* is ignored.

        Raises
        ------
        ValueError
            If both or neither of mean_motion / semi_major_axis are provided,
            or if the inclination is too close to 0° or 180° to determine RAAN.
        """
        import warnings

        if (mean_motion is None) == (semi_major_axis is None):
            raise ValueError("Provide exactly one of mean_motion or semi_major_axis.")

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        inc     = np.deg2rad(inclination)
        omega   = np.deg2rad(argp)
        e       = float(eccentricity)

        # ── ECI position at epoch ────────────────────────────────────────────
        r_ecef = lla_to_ecef(lat_rad, lon_rad, alt)
        r_eci  = ecef_to_eci(r_ecef, jd)
        r_mag  = np.linalg.norm(r_eci)

        # ── Semi-major axis and mean motion ──────────────────────────────────
        _two_pi = 2.0 * np.pi
        if mean_motion is not None:
            mm      = float(mean_motion)
            n_rad_s = mm * _two_pi / 86400.0
            a       = (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0)
        else:
            a       = float(semi_major_axis)
            n_rad_s = (MU_EARTH / a**3) ** 0.5
            mm      = n_rad_s * 86400.0 / _two_pi

        # ── Argument of latitude u = argp + nu ───────────────────────────────
        # From the ECI position: rz = r_mag * sin(u) * sin(i)
        sin_i = np.sin(inc)
        if abs(sin_i) < 1e-10:
            raise ValueError(
                "Inclination too close to 0° or 180° to determine RAAN from position."
            )

        sin_u   = np.clip(r_eci[2] / (r_mag * sin_i), -1.0, 1.0)
        u_base  = np.arcsin(sin_u)   # principal value in [-π/2, π/2]

        # ascending → cos(u) > 0  →  u in (-π/2, π/2)  →  use u_base directly
        # descending → cos(u) < 0  →  u in  (π/2, 3π/2) →  supplement
        u = u_base % _two_pi if ascending else (np.pi - u_base) % _two_pi

        # ── True anomaly ─────────────────────────────────────────────────────
        nu = (u - omega) % _two_pi

        # ── Consistency check: does |r| match the orbital shape? ─────────────
        p           = a * (1.0 - e**2)
        r_expected  = p / (1.0 + e * np.cos(nu))
        if abs(r_expected - r_mag) / r_mag > 0.01:
            warnings.warn(
                f"Position radius {r_mag / 1e3:.1f} km differs from orbital shape "
                f"({r_expected / 1e3:.1f} km expected). "
                "Verify that semi_major_axis/mean_motion, eccentricity, and argp are consistent.",
                UserWarning, stacklevel=2,
            )

        # ── RAAN ─────────────────────────────────────────────────────────────
        # From: rx = r*(cos(Ω)*cos(u) - sin(Ω)*cos(i)*sin(u))
        #        ry = r*(sin(Ω)*cos(u) + cos(Ω)*cos(i)*sin(u))
        # → [rx,ry] = R(Ω) @ [cos(u), cos(i)*sin(u)]
        # → Ω = atan2(ry,rx) - atan2(cos(i)*sin(u), cos(u))
        raan = (
            np.arctan2(r_eci[1], r_eci[0])
            - np.arctan2(np.cos(inc) * np.sin(u), np.cos(u))
        ) % _two_pi

        # ── Mean anomaly ─────────────────────────────────────────────────────
        if e < 1e-10:
            M = nu % _two_pi
        else:
            E = 2.0 * np.arctan2(
                np.sqrt(1.0 - e) * np.sin(nu / 2.0),
                np.sqrt(1.0 + e) * np.cos(nu / 2.0),
            )
            M = (E - e * np.sin(E)) % _two_pi

        # ── Assemble TLE ─────────────────────────────────────────────────────
        epoch_year, epoch_day = _jd_to_epoch(jd)

        t = TLE(
            name=name,
            norad_id=norad_id,
            classification="U",
            intl_designator="",
            epoch_year=epoch_year,
            epoch_day=epoch_day,
            ndot=ndot,
            nddot=0.0,
            bstar=bstar,
            element_set=0,
            inclination=inc,
            raan=raan,
            eccentricity=e,
            argp=omega,
            mean_anomaly=M,
            mean_motion=mm,
            rev_number=0,
            epoch_jd=jd,
            semi_major_axis=a,
            catalog_id=f"{norad_id:05d}",
        )

        return cls(tle=t)

    def __repr__(self) -> str:
        return (
            f"Satellite({self.name!r}, NORAD={self.norad_id}, "
            f"a={self.semi_major_axis / 1e3:.1f} km, "
            f"e={self.eccentricity:.4f}, "
            f"i={np.rad2deg(self.inclination):.2f}°, "
            f"{self.orbit_type})"
        )
