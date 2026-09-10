# Runtime projection from measured replications

Measured complete replications: 240. The projection covers 12 requested case/n groups.

- Q=20: **13.0 minutes (0.22 hours)** for all fits; 0.0 minutes remaining if these checkpoints can be reused.
- Q=200: **129.5 minutes (2.16 hours)** for all fits; 116.6 minutes remaining if these checkpoints can be reused.

These are rough sequential wall-time estimates, not precision bounds. They require the same network, epoch cap, patience, optimizer, thread count, density routine, and auxiliary-projection settings as the measured fits. A deliberately shortened smoke run is not representative of a longer training configuration. Early stopping, process startup, and machine load can alter the result.

runtime_projection.csv records each estimate and its source. Exact case/n mean timings are used when available. Missing n values use linear interpolation between measured n values, or proportional-to-n extrapolation from the nearest endpoint. Cases without measurements use pooled observed cases. Extrapolation from a single small n can be inaccurate. Report/figure generation overhead is excluded. Remaining-time estimates presume every measured row is a reusable compatible checkpoint.

Q=200 is projected only; this function never launches fits.
