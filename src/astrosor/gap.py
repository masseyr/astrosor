from skyfield.api import load, wgs84
from datetime import timedelta

# 1. Define your timeframe
ts = load.timescale()
start_time = ts.utc(2026, 3, 10, 0, 0)
end_time = ts.utc(2026, 3, 11, 0, 0)

def calculate_gaps(target_pos, sensors, start_t, end_t):
    all_access_windows = []
    
    for sensor in sensors:
        # Find intervals where the satellite is above the horizon (e.g., 30 degrees)
        t, events = sensor.find_events(target_pos, start_t, end_t, altitude_degrees=30.0)
        
        # skyfield events: 0=rise, 1=culminate, 2=set
        # Logic to pair (rise, set) into [start, end] tuples
        windows = [(t[i].tt, t[i+2].tt) for i in range(0, len(events)-2, 3)]
        all_access_windows.extend(windows)

    # 2. Sort and Merge Intervals
    if not all_access_windows:
        return [(start_t.tt, end_t.tt)] # 100% gap
        
    all_access_windows.sort()
    merged = []
    curr_start, curr_end = all_access_windows[0]
    
    for next_start, next_end in all_access_windows[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))

    # 3. Calculate Gaps between merged windows
    gaps = []
    # Gap before first window
    if merged[0][0] > start_t.tt:
        gaps.append(merged[0][0] - start_t.tt)
    
    # Gaps between windows
    for i in range(len(merged) - 1):
        gap_dur = merged[i+1][0] - merged[i][1]
        gaps.append(gap_dur * 24 * 3600) # Convert JD days to seconds
        
    return gaps # List of instantaneous gap durations in seconds