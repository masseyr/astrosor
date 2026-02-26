Astro coordinate transform library






## Tests

| Module | Tests | Coverage highlights |
|--------|------:|---------------------|
| **utils** | 22 | normalize, rotation matrices, JD, GMST, ECI↔ECEF, LLA↔ECEF, constants |
| **frames** | 42 | UNW/TNW/ECR DCMs, all pairwise roundtrips, transport theorem, covariance, unified API |
| **orbits** | 12 | Keplerian↔ECI, conservation laws, J2, propagation |
| **coverage** | 12 | Earth intersection, footprints, access geometry, ground trace |
| **sun** | 22 | Ephemeris, exclusion, eclipse (cylindrical + conical), phase angle, illumination |
| **moon** | 17 | Ephemeris, exclusion, illumination/phase, full/new Moon validation |
| **exclusion** | 9 | Combined checks, availability timeline, target observability |
| **tle** | 13 | Parsing (3-line, 2-line, batch), epoch state, propagation |
| **sensor** | 16 | ECEF/ECI geometry, azel, visual magnitude, radar + optical visibility |
| **scheduler** | 21 | Urgency (formula verification), scoring, visibility windows, greedy scheduling (end-to-end, multi-sat, priority dominance, no-overlap, empty schedule), format output |