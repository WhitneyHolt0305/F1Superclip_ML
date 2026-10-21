# estimate_costs.py  (OpenF1 version)
# Goal: get rough numbers for the loss function.
#   Miss cost        = time lost when a car superclips on a straight
#   False alarm cost = time lost when a car lifts and coasts
#
# Setup (run once):  uv add polars pyarrow numpy requests
# Data: OpenF1 (free, no key). Limit = 3 requests per second.

import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

# ---------- Settings (change these) ----------
YEAR = 2026
SESSION_NAMES = ["Qualifying", "Race"]
MAX_SESSIONS = 2          # start small to test; set to None to use every session
FULL_THROTTLE = 99        # throttle % that counts as "flat out"
SUPERCLIP_DROP = 5        # km/h drop that counts as superclipping
MIN_STRAIGHT_M = 200      # ignore flat-out bits shorter than this
COAST_THROTTLE = 10       # throttle % below this (no brake) = coasting
MIN_COAST_S = 0.5         # ignore coasts shorter than this (tiny lifts in corners)
SLOW_LAP_CUTOFF = 1.07    # drop laps slower than 107% of the session's fastest

API = "https://api.openf1.org/v1"
CACHE = Path("openf1_cache")
CACHE.mkdir(exist_ok=True)
DATE_FMT = "%Y-%m-%dT%H:%M:%S%.f%:z"


# ---------- Download helper (saves a copy so you only download once) ----------
def get_data(endpoint, **params):
    name = endpoint + "_" + "_".join(f"{k}-{v}" for k, v in params.items())
    file = CACHE / f"{name}.parquet"
    if file.exists():
        return pl.read_parquet(file)

    for attempt in range(5):
        resp = requests.get(f"{API}/{endpoint}", params=params, timeout=60)
        if resp.status_code == 429:          # too many requests: wait and retry
            time.sleep(2 + attempt)
            continue
        if resp.status_code == 404:          # no data (e.g. cancelled session)
            time.sleep(0.4)
            return pl.DataFrame()
        resp.raise_for_status()
        break
    time.sleep(0.4)                          # stay under 3 requests per second

    df = pl.DataFrame(resp.json(), infer_schema_length=None)
    df.write_parquet(file)
    return df


# ---------- Helper: find stretches where a condition is True ----------
def find_runs(mask):
    """Return (start, end) row positions for each run of True values."""
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


# ---------- Miss cost: time lost on superclipping straights ----------
def straight_rows(lap):
    speed = lap["speed"].to_numpy()          # km/h
    dist = lap["distance_m"].to_numpy()      # meters
    secs = lap["secs"].to_numpy()            # seconds since lap start
    flat = (lap["throttle"].to_numpy() >= FULL_THROTTLE) & (lap["brake"].to_numpy() == 0)

    rows = []
    for s, e in find_runs(flat):
        if dist[e] - dist[s] < MIN_STRAIGHT_M:
            continue
        peak = s + int(np.argmax(speed[s:e + 1]))   # where top speed happens
        drop = speed[peak] - speed[e]               # speed lost after the peak
        # Time lost = actual time after peak - time if the car held peak speed
        held_time = (dist[e] - dist[peak]) / (speed[peak] / 3.6)
        rows.append({
            "straight_m": dist[e] - dist[s],
            "speed_drop_kmh": drop,
            "superclip": int(drop >= SUPERCLIP_DROP),
            "time_lost_s": (secs[e] - secs[peak]) - held_time,
        })
    return rows


# ---------- False alarm cost: time lost while coasting ----------
def coast_rows(lap):
    speed = lap["speed"].to_numpy()
    dist = lap["distance_m"].to_numpy()
    secs = lap["secs"].to_numpy()
    coast = (lap["throttle"].to_numpy() < COAST_THROTTLE) & (lap["brake"].to_numpy() == 0)

    rows = []
    for s, e in find_runs(coast):
        if e == s or speed[s] == 0:
            continue  # too short, or car stopped
        if secs[e] - secs[s] < MIN_COAST_S:
            continue  # just a tiny lift, not a real lift-and-coast
        held_time = (dist[e] - dist[s]) / (speed[s] / 3.6)
        rows.append({"coast_time_lost_s": (secs[e] - secs[s]) - held_time})
    return rows


# ---------- Match telemetry to laps and add distance ----------
def add_laps(car, laps):
    """Give each telemetry row a lap number, then add time and distance within the lap."""
    car = (car.with_columns(pl.col("date").str.to_datetime(DATE_FMT, time_zone="UTC"))
              .sort("date"))
    laps = (laps.with_columns(pl.col("date_start").str.to_datetime(DATE_FMT, time_zone="UTC"))
                .sort("date_start"))

    joined = car.join_asof(laps, left_on="date", right_on="date_start")
    joined = joined.filter(
        pl.col("date") < pl.col("date_start")
        + pl.duration(milliseconds=pl.col("lap_duration") * 1000)
    )
    return (joined
            .with_columns(
                ((pl.col("date") - pl.col("date").first())
                 .dt.total_milliseconds() / 1000).over("lap_number").alias("secs"))
            .with_columns(
                pl.col("secs").diff().fill_null(0).over("lap_number").alias("dt"))
            .with_columns(
                (pl.col("speed") / 3.6 * pl.col("dt")).cum_sum()
                .over("lap_number").alias("distance_m")))


# ---------- Pick sessions ----------
sessions = (get_data("sessions", year=YEAR)
            .filter(pl.col("session_name").is_in(SESSION_NAMES))
            .sort("date_start"))
if MAX_SESSIONS:
    sessions = sessions.head(MAX_SESSIONS)
print(sessions.select("session_key", "country_name", "session_name"))


# ---------- Loop over sessions and drivers ----------
straights, coasts = [], []

for ses in sessions.iter_rows(named=True):
    key, label = ses["session_key"], f"{ses['country_name']} {ses['session_name']}"
    all_laps = get_data("laps", session_key=key)
    if all_laps.is_empty():
        print(f"Skipped {label}: no lap data (cancelled or not run yet)")
        continue

    # Keep clean, timed laps only (no lap 1, no pit-out laps, not too slow)
    all_laps = all_laps.filter(pl.col("lap_duration").is_not_null()
                               & (pl.col("lap_number") > 1)
                               & (pl.col("is_pit_out_lap") == False))
    if all_laps.is_empty():
        print(f"Skipped {label}: no lap data")
        continue
    fastest = all_laps["lap_duration"].min()
    all_laps = all_laps.filter(pl.col("lap_duration") <= fastest * SLOW_LAP_CUTOFF)

    for drv in all_laps["driver_number"].unique().to_list():
        laps = all_laps.filter(pl.col("driver_number") == drv).select(
            "lap_number", "date_start", "lap_duration")
        print(f"  {label}: downloading driver {drv}...")
        car = get_data("car_data", session_key=key, driver_number=drv)
        if car.is_empty():
            continue
        tel = add_laps(car.select("date", "speed", "throttle", "brake"), laps)

        for lap in tel.partition_by("lap_number"):
            info = {"session": ses["session_name"], "event": ses["country_name"],
                    "driver": drv, "lap": lap["lap_number"][0]}
            straights += [{**info, **r} for r in straight_rows(lap)]
            coasts += [{**info, **r} for r in coast_rows(lap)]

    print(f"Done: {label}")

if not straights:
    raise SystemExit("No straights found. Check the messages above.")

straights = pl.DataFrame(straights)
coasts = pl.DataFrame(coasts)


# ---------- Summary: the numbers you need ----------
miss_cost = (straights.filter(pl.col("superclip") == 1)
             .group_by("session")
             .agg(pl.col("time_lost_s").mean().alias("miss_cost_s"),
                  pl.len().alias("n_superclips")))

false_alarm_cost = (coasts.group_by("session")
                    .agg(pl.col("coast_time_lost_s").mean().alias("false_alarm_cost_s"),
                         pl.len().alias("n_coasts")))

costs = (miss_cost.join(false_alarm_cost, on="session")
         .with_columns(
             (pl.col("miss_cost_s") / pl.col("false_alarm_cost_s")).alias("times_worse"),
             (pl.col("false_alarm_cost_s")
              / (pl.col("miss_cost_s") + pl.col("false_alarm_cost_s"))).alias("cutoff"),
         ))

print(costs)
print("Share of straights with superclipping:")
print(straights.group_by("session").agg(pl.col("superclip").mean()))

# Save so you don't have to rerun
straights.write_parquet("straights.parquet")
coasts.write_parquet("coasts.parquet")
