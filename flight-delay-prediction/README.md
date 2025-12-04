# Flight Delay & Cancellation Prediction (2019–2023)

**Author:** Bri Martinez

This repository contains the code and artifacts for a leakage-safe, pre-departure prediction pipeline for U.S. flight delays and cancellations. It implements the final project described in previous proposal/progress report: calibrated classifiers, ablation studies (weather vs schedule-only), temporal evaluation, and auto-generated Q&A summaries.

## Project Description

- **Goal:** Estimate the probability that a flight will be delayed (≥15 min) or cancelled using route, schedule, and weather features available *before* departure.
- **Why it matters:** Delays/cancellations impose significant costs; calibrated probabilities enable better staffing, buffering, and proactive passenger alerts.
- **Scope:** BTS On-Time Performance (2019–2023) as primary data; NOAA METAR/ASOS for weather enrichment; strict anti-leakage feature view.

## Research Questions

1. **Does adding weather improve discrimination and calibration** over schedule-only baselines?
2. **Which factors** (schedule, route, weather) **contribute most** to predicted risk?
3. **How stable** are results across **pre-/during-/post-COVID** periods?
4. **What operating threshold** delivers useful alerts at modest volume (top-k or cost-based)?
