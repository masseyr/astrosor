"""
orbital_frames.scheduler — Ground Sensor Task Scheduler
=========================================================

Greedy priority-based task scheduler for ground-based radar and optical
sensors observing satellites from TLEs.

Pipeline
--------
1. **TLE Catalog Ingest**: Parse TLEs with assigned priority and revisit rates.
2. **Visibility Pre-computation**: Propagate each satellite from its TLE epoch
   to the simulation window, evaluate sensor visibility at each time step.
3. **Opportunity Extraction**: Build per-satellite visibility windows (rise/set).
4. **Priority Scoring**: Dynamic priority = base priority × revisit urgency,
   where urgency increases as time since last observation exceeds desired
   revisit interval.
5. **Greedy Scheduling**: At each decision epoch, select the highest-priority
   observable target, allocate a task (dwell time), and advance.

Priority Model
--------------
The scheduling priority for satellite *i* at time *t* is::

    score_i(t) = base_priority_i × urgency_i(t) × elevation_bonus_i(t)

where::

    urgency_i(t) = 1 + max(0, (t − t_last_obs_i) / revisit_interval_i − 1)²

This gives urgency=1 when the satellite was recently observed, and
grows quadratically once the revisit interval is exceeded, ensuring
overdue satellites rapidly dominate the schedule.

The elevation bonus favors observations near culmination::

    elevation_bonus = 1 + 0.5 × sin(elevation)
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field as dc_field

from .tle import TLE, propagate_tle
from .sensor import GroundSensor, check_visibility, topocentric_azel
from .utils import eci_to_ecef


# ════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SatelliteTask:
    """A satellite in the scheduling catalog."""
    tle: TLE
    base_priority: int = 5          # 1 = highest, 10 = lowest
    revisit_interval: float = 3600.0  # desired revisit [s]
    rcs: float = 1.0                # radar cross-section [m²]
    intrinsic_mag: float = 6.0      # optical reference magnitude
    last_observed: float = -np.inf  # JD of last observation


@dataclass
class VisibilityWindow:
    """A contiguous visibility window for one satellite."""
    sat_index: int
    norad_id: int
    name: str
    start_jd: float
    end_jd: float
    peak_el: float          # peak elevation [rad]
    peak_el_jd: float       # JD at peak elevation
    min_range: float        # closest range [m]
    mean_az: float          # mean azimuth [rad]
    # For optical: worst-case magnitude during window
    brightest_mag: float = 99.0


@dataclass
class ScheduledTask:
    """A scheduled observation task."""
    sat_index: int
    norad_id: int
    name: str
    start_jd: float
    end_jd: float
    dwell_time: float       # [s]
    score: float            # scheduling score at decision time
    elevation: float        # elevation at task start [rad]
    azimuth: float          # azimuth at task start [rad]
    slant_range: float      # range at task start [m]
    visual_mag: float       # estimated vmag (optical) or NaN (radar)
    urgency: float          # urgency factor at decision time
    time_since_last: float  # seconds since last obs of this satellite


@dataclass
class ScheduleResult:
    """Complete scheduling output."""
    tasks: list               # list of ScheduledTask
    windows: list             # list of VisibilityWindow
    unscheduled_windows: list  # windows that couldn't be scheduled
    stats: dict               # summary statistics


# ════════════════════════════════════════════════════════════════════════════
#  Visibility Window Computation
# ════════════════════════════════════════════════════════════════════════════

def compute_visibility_windows(
    sensor: GroundSensor,
    satellites: list[SatelliteTask],
    jd_start: float,
    jd_end: float,
    dt_step: float = 30.0,
) -> list[VisibilityWindow]:
    """Pre-compute all visibility windows for all satellites over the sim span.

    Parameters
    ----------
    sensor : GroundSensor — the observing sensor
    satellites : list[SatelliteTask] — catalog of satellites
    jd_start : float — simulation start (Julian Date)
    jd_end : float — simulation end (Julian Date)
    dt_step : float — visibility sampling step [s]

    Returns
    -------
    windows : list[VisibilityWindow] — all windows, sorted by start time
    """
    duration_s = (jd_end - jd_start) * 86400.0
    n_steps = int(np.ceil(duration_s / dt_step)) + 1
    times_jd = jd_start + np.arange(n_steps) * dt_step / 86400.0

    all_windows = []

    for sat_idx, sat in enumerate(satellites):
        vis_flags = np.zeros(n_steps, dtype=bool)
        elevations = np.full(n_steps, -np.pi / 2)
        azimuths = np.zeros(n_steps)
        ranges = np.full(n_steps, np.inf)
        mags = np.full(n_steps, 99.0)

        for k, jd in enumerate(times_jd):
            try:
                r_sat = propagate_tle(sat.tle, jd)[0]
            except Exception:
                continue

            vis = check_visibility(sensor, r_sat, jd, sat.rcs, sat.intrinsic_mag)
            vis_flags[k] = vis["visible"]
            elevations[k] = vis["el"]
            azimuths[k] = vis["az"]
            ranges[k] = vis["range"]
            if "visual_mag" in vis:
                mags[k] = vis["visual_mag"]

        # Extract contiguous windows
        diff = np.diff(vis_flags.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if vis_flags[0]:
            starts = np.concatenate([[0], starts])
        if vis_flags[-1]:
            ends = np.concatenate([ends, [n_steps - 1]])

        for s, e in zip(starts, ends):
            seg = slice(s, e + 1)
            peak_idx = s + np.argmax(elevations[seg])
            all_windows.append(VisibilityWindow(
                sat_index=sat_idx,
                norad_id=sat.tle.norad_id,
                name=sat.tle.name,
                start_jd=float(times_jd[s]),
                end_jd=float(times_jd[e]),
                peak_el=float(elevations[peak_idx]),
                peak_el_jd=float(times_jd[peak_idx]),
                min_range=float(np.min(ranges[seg])),
                mean_az=float(np.mean(azimuths[seg])),
                brightest_mag=float(np.min(mags[seg])),
            ))

    # Sort by start time
    all_windows.sort(key=lambda w: w.start_jd)
    return all_windows


# ════════════════════════════════════════════════════════════════════════════
#  Priority Scoring
# ════════════════════════════════════════════════════════════════════════════

def compute_urgency(
    time_since_last_obs: float,
    revisit_interval: float,
) -> float:
    """Compute urgency factor based on revisit requirement.

    Returns 1.0 when recently observed, grows quadratically when overdue.

    Parameters
    ----------
    time_since_last_obs : float — seconds since last observation
    revisit_interval : float — desired revisit interval [s]

    Returns
    -------
    urgency : float ≥ 1.0
    """
    if revisit_interval <= 0:
        return 1.0
    ratio = time_since_last_obs / revisit_interval
    if ratio <= 1.0:
        return 1.0
    return 1.0 + (ratio - 1.0) ** 2


def compute_score(
    base_priority: int,
    urgency: float,
    elevation: float,
) -> float:
    """Compute scheduling score (higher = more desirable).

    Parameters
    ----------
    base_priority : int — 1 (highest) to 10 (lowest)
    urgency : float — revisit urgency factor (≥1.0)
    elevation : float — current elevation [rad]

    Returns
    -------
    score : float — non-negative scheduling score
    """
    # Invert priority so 1 → highest score
    priority_factor = 11.0 - float(base_priority)

    # Elevation bonus: prefer higher passes
    el_bonus = 1.0 + 0.5 * np.sin(max(elevation, 0.0))

    return priority_factor * urgency * el_bonus


# ════════════════════════════════════════════════════════════════════════════
#  Greedy Scheduler
# ════════════════════════════════════════════════════════════════════════════

def schedule_greedy(
    sensor: GroundSensor,
    satellites: list[SatelliteTask],
    jd_start: float,
    jd_end: float,
    dwell_time: float = 10.0,
    slew_time: float = 5.0,
    dt_decision: float = 10.0,
    dt_visibility: float = 30.0,
    windows: list[VisibilityWindow] | None = None,
) -> ScheduleResult:
    """Run the greedy priority-based task scheduler.

    At each decision epoch the scheduler:
      1. Identifies all currently-visible satellites
      2. Scores each by  priority × urgency × elevation_bonus
      3. Selects the highest-scoring target
      4. Allocates a dwell, marks the satellite as observed
      5. Advances by dwell_time + slew_time

    Parameters
    ----------
    sensor : GroundSensor — the observing sensor
    satellites : list[SatelliteTask] — catalog with priorities/revisit rates
    jd_start : float — schedule start (Julian Date)
    jd_end : float — schedule end (Julian Date)
    dwell_time : float — observation dwell per task [s]
    slew_time : float — sensor slew/settle time between tasks [s]
    dt_decision : float — decision epoch spacing when idle [s]
    dt_visibility : float — visibility pre-computation step [s]
    windows : list or None — pre-computed windows (if None, computed internally)

    Returns
    -------
    ScheduleResult with tasks, windows, unscheduled windows, and statistics
    """
    # ── Pre-compute visibility windows ──
    if windows is None:
        windows = compute_visibility_windows(
            sensor, satellites, jd_start, jd_end, dt_visibility
        )

    # ── Initialize tracking state ──
    last_obs_jd = {}  # norad_id → JD of last observation
    for sat in satellites:
        if sat.last_observed > 0:
            last_obs_jd[sat.tle.norad_id] = sat.last_observed
        else:
            # Treat as never observed (infinite urgency at start)
            last_obs_jd[sat.tle.norad_id] = jd_start - sat.revisit_interval / 86400.0 * 2.0

    tasks = []
    scheduled_window_ids = set()

    # ── Main scheduling loop ──
    current_jd = jd_start
    task_interval_jd = (dwell_time + slew_time) / 86400.0
    decision_step_jd = dt_decision / 86400.0

    while current_jd < jd_end:
        # Find all windows active at current_jd
        candidates = []
        for w_idx, w in enumerate(windows):
            if w_idx in scheduled_window_ids:
                continue
            if w.start_jd <= current_jd <= w.end_jd:
                sat_idx = w.sat_index
                sat = satellites[sat_idx]
                nid = w.norad_id

                # Compute current elevation
                try:
                    r_sat = propagate_tle(sat.tle, current_jd)[0]
                    r_sat_ecef = eci_to_ecef(r_sat, current_jd)
                    az, el, rng = topocentric_azel(
                        sensor.ecef(), sensor.lat, sensor.lon, r_sat_ecef
                    )
                except Exception:
                    continue

                if el < sensor.min_elevation:
                    continue

                # Re-verify visibility at decision epoch
                vis = check_visibility(sensor, r_sat, current_jd, sat.rcs, sat.intrinsic_mag)
                if not vis["visible"]:
                    continue

                # Score
                t_since = (current_jd - last_obs_jd.get(nid, jd_start - 1.0)) * 86400.0
                urg = compute_urgency(t_since, sat.revisit_interval)
                score = compute_score(sat.base_priority, urg, el)

                vmag = vis.get("visual_mag", np.nan)

                candidates.append({
                    "w_idx": w_idx,
                    "sat_idx": sat_idx,
                    "nid": nid,
                    "name": w.name,
                    "score": score,
                    "az": az,
                    "el": el,
                    "range": rng,
                    "vmag": vmag,
                    "urgency": urg,
                    "t_since": t_since,
                })

        if not candidates:
            # No targets visible — advance to next decision epoch
            current_jd += decision_step_jd
            continue

        # ── Select best candidate ──
        best = max(candidates, key=lambda c: c["score"])

        task = ScheduledTask(
            sat_index=best["sat_idx"],
            norad_id=best["nid"],
            name=best["name"],
            start_jd=current_jd,
            end_jd=current_jd + dwell_time / 86400.0,
            dwell_time=dwell_time,
            score=best["score"],
            elevation=best["el"],
            azimuth=best["az"],
            slant_range=best["range"],
            visual_mag=best["vmag"],
            urgency=best["urgency"],
            time_since_last=best["t_since"],
        )
        tasks.append(task)

        # Update state
        last_obs_jd[best["nid"]] = current_jd
        scheduled_window_ids.add(best["w_idx"])

        # Advance past dwell + slew
        current_jd += task_interval_jd

    # ── Identify unscheduled windows ──
    unsched = [w for i, w in enumerate(windows) if i not in scheduled_window_ids]

    # ── Statistics ──
    duration_h = (jd_end - jd_start) * 24.0
    unique_sats = len(set(t.norad_id for t in tasks))
    total_sats = len(satellites)

    obs_counts = {}
    for t in tasks:
        obs_counts[t.norad_id] = obs_counts.get(t.norad_id, 0) + 1

    # Revisit compliance
    revisit_met = 0
    for sat in satellites:
        nid = sat.tle.norad_id
        if nid in obs_counts:
            expected = max(1, int(duration_h * 3600 / max(sat.revisit_interval, 1)))
            if obs_counts[nid] >= expected * 0.8:  # 80% threshold
                revisit_met += 1

    stats = {
        "total_tasks": len(tasks),
        "total_windows": len(windows),
        "unscheduled_windows": len(unsched),
        "unique_satellites_observed": unique_sats,
        "total_satellites": total_sats,
        "coverage_fraction": unique_sats / max(total_sats, 1),
        "schedule_duration_hours": duration_h,
        "sensor_utilization": len(tasks) * (dwell_time + slew_time) / (duration_h * 3600) if duration_h > 0 else 0,
        "mean_score": float(np.mean([t.score for t in tasks])) if tasks else 0.0,
        "mean_elevation_deg": float(np.rad2deg(np.mean([t.elevation for t in tasks]))) if tasks else 0.0,
        "observation_counts": obs_counts,
        "revisit_compliance_fraction": revisit_met / max(total_sats, 1),
    }

    return ScheduleResult(
        tasks=tasks,
        windows=windows,
        unscheduled_windows=unsched,
        stats=stats,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Schedule Formatting
# ════════════════════════════════════════════════════════════════════════════

def format_schedule(result: ScheduleResult, sensor: GroundSensor) -> str:
    """Format a ScheduleResult as a human-readable report.

    Parameters
    ----------
    result : ScheduleResult
    sensor : GroundSensor

    Returns
    -------
    report : str
    """
    lines = []
    lines.append("=" * 78)
    lines.append(f"  TASK SCHEDULE — {sensor.name} ({sensor.sensor_type.upper()})")
    lines.append(f"  Location: {np.rad2deg(sensor.lat):.4f}°N, {np.rad2deg(sensor.lon):.4f}°E, {sensor.alt:.0f}m")
    lines.append("=" * 78)

    s = result.stats
    lines.append(f"\n  Duration:         {s['schedule_duration_hours']:.1f} hours")
    lines.append(f"  Total tasks:      {s['total_tasks']}")
    lines.append(f"  Visibility windows: {s['total_windows']}")
    lines.append(f"  Unique sats:      {s['unique_satellites_observed']} / {s['total_satellites']}")
    lines.append(f"  Sensor util:      {s['sensor_utilization']*100:.1f}%")
    lines.append(f"  Mean elevation:   {s['mean_elevation_deg']:.1f}°")
    lines.append(f"  Revisit compliance: {s['revisit_compliance_fraction']*100:.0f}%")

    if result.tasks:
        lines.append(f"\n  {'#':>4s}  {'NORAD':>6s}  {'Name':<20s}  {'Az':>6s}  {'El':>5s}  "
                     f"{'Range':>8s}  {'Score':>6s}  {'Urg':>5s}  {'T_since':>7s}")
        lines.append("  " + "-" * 74)

        for k, t in enumerate(result.tasks):
            az_d = np.rad2deg(t.azimuth)
            el_d = np.rad2deg(t.elevation)
            rng_km = t.slant_range / 1e3
            t_since_m = t.time_since_last / 60.0
            name = t.name[:20] if t.name else f"SAT-{t.norad_id}"
            lines.append(
                f"  {k+1:4d}  {t.norad_id:6d}  {name:<20s}  {az_d:6.1f}  {el_d:5.1f}  "
                f"{rng_km:8.0f}  {t.score:6.1f}  {t.urgency:5.2f}  {t_since_m:7.1f}m"
            )

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)
