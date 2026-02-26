"""
orbital_frames — Space Situational Awareness Coordinate & Scheduling Library
=============================================================================

Pure-NumPy library for coordinate frame transforms (ECI/ECR/UNW/TNW),
solar/lunar ephemeris and exclusion analysis, ground sensor models,
TLE propagation, and priority-based observation task scheduling.

Modules: frames, orbits, coverage, sun, moon, exclusion, tle, sensor, scheduler
"""

from .frames import (
    eci_to_ecr_matrix, ecr_to_eci_matrix, eci_to_ecr, ecr_to_eci,
    state_eci_to_ecr, state_ecr_to_eci,
    transform_covariance_eci_to_ecr, transform_covariance_ecr_to_eci,
    eci_to_unw_matrix, unw_to_eci_matrix, eci_to_unw, unw_to_eci,
    state_eci_to_unw, transform_covariance_eci_to_unw, transform_covariance_unw_to_eci,
    eci_to_tnw_matrix, tnw_to_eci_matrix, eci_to_tnw, tnw_to_eci,
    state_eci_to_tnw, transform_covariance_eci_to_tnw, transform_covariance_tnw_to_eci,
    unw_to_tnw, tnw_to_unw,
    ecr_to_unw, unw_to_ecr, ecr_to_tnw, tnw_to_ecr,
    transform_covariance_ecr_to_unw, transform_covariance_unw_to_ecr,
    transform_covariance_ecr_to_tnw, transform_covariance_tnw_to_ecr,
    transform_covariance_unw_to_tnw, transform_covariance_tnw_to_unw,
    get_dcm, transform, transform_covariance, FRAMES,
)
from .orbits import (
    keplerian_to_eci, eci_to_keplerian, propagate_kepler,
    compute_orbital_period, compute_mean_motion,
)
from .coverage import (
    sensor_footprint_unw, sensor_footprint_tnw,
    ground_trace, coverage_windows, access_geometry, earth_intersection,
)
from .sun import (
    sun_position_eci, sun_direction_eci, sun_distance, solar_declination_ra,
    sun_angle, check_solar_exclusion, solar_exclusion_windows,
    eclipse_cylindrical, eclipse_conical, eclipse_intervals,
    solar_phase_angle, is_target_illuminated, sep_angle,
    AU, R_SUN, SOLAR_FLUX_1AU,
)
from .moon import (
    moon_position_eci, moon_direction_eci, moon_distance,
    moon_angle, moon_angular_radius, check_lunar_exclusion, lunar_exclusion_windows,
    moon_phase_angle, moon_illumination_fraction, moon_age_days, moon_phase_name,
    sun_moon_angle, moon_earth_shadow, R_MOON, MEAN_EARTH_MOON_DIST,
)
from .exclusion import (
    check_all_exclusions, availability_timeline, target_observability,
)
from .tle import (
    TLE, parse_tle, parse_tle_batch, tle_epoch_state, propagate_tle,
)
from .sensor import (
    GroundSensor, check_visibility, topocentric_azel,
    sun_elevation_at_sensor, estimate_visual_magnitude,
)
from .scheduler import (
    SatelliteTask, VisibilityWindow, ScheduledTask, ScheduleResult,
    compute_visibility_windows, compute_urgency, compute_score,
    schedule_greedy, format_schedule,
)
from .utils import (
    normalize, rotation_matrix_axis_angle,
    eci_to_ecef, ecef_to_eci, ecef_to_lla, lla_to_ecef,
    julian_date, gmst,
    MU_EARTH, R_EARTH, J2, OMEGA_EARTH, F_EARTH, E2_EARTH,
)

__version__ = "4.0.0"
