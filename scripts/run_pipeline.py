from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths / settings
# -----------------------------------------------------------------------------
INPUTS_DIR = Path("Inputs")
OUTPUTS_DIR = Path("Outputs")

# Raw Cerema export (semicolon-separated, with '#' metadata lines)
MAIN_TIMESERIES_FILE = INPUTS_DIR / "data_01052021_170124.txt"

# (Optional) not used yet in this script, kept for later expansion
WATER_LEVEL_FILE = INPUTS_DIR / "water_level_P115.txt"

# You requested 30-min resampling
RESAMPLE_FREQ = "30min"


# -----------------------------------------------------------------------------
# I/O + preprocessing
# -----------------------------------------------------------------------------
def load_main_timeseries(path: Path) -> pd.DataFrame:
    """
    Reads the raw Cerema export:
    - metadata lines start with '#'
    - separator is ';'
    - columns: time; pluvio1; pluvio7; flowrate; waterlevel; storedvolume
    """
    df = pd.read_csv(
        path,
        sep=";",
        comment="#",  # skips metadata lines
        header=None,
        names=["time", "pluvio1", "pluvio7", "flowrate", "waterlevel", "storedvolume"],
        encoding="utf-8",
    )

    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("time").sort_index()

    # Replace known missing codes with NaN
    df.replace([-6999, -7999, -9999, -9.999], np.nan, inplace=True)

    # Some sensors can appear negative; take absolute values
    for c in ["flowrate", "waterlevel", "storedvolume"]:
        df[c] = df[c].abs()

    return df


def resample_30min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to 30 min:
    - rainfall is summed
    - sensors are averaged
    """
    return df.resample(RESAMPLE_FREQ).agg(
        {
            "pluvio1": "sum",
            "pluvio7": "sum",
            "flowrate": "mean",
            "waterlevel": "mean",
            "storedvolume": "mean",
        }
    )


# -----------------------------------------------------------------------------
# CEREMA-style rain event detection (rain-only)
# Based on your internship Rain_event output parameters:
# Pdt=1 min, I_seuil=12 mm/h, Pdt_fin=60 min, H_fin=0.2 mm, H_min=0.2 mm
# -----------------------------------------------------------------------------
def detect_rain_events_cerema(
    rain_mm: pd.Series,
    pdt_min: float = 1.0,
    I_seuil_mm_h: float = 12.0,
    pdt_fin_min: float = 60.0,
    H_fin_mm: float = 0.2,
    H_min_mm: float = 0.2,
) -> pd.DataFrame:
    """
    Reproduces the CEREMA Rain_event logic for rain-only event detection
    (no runoff tests).

    Event starts when intensity >= I_seuil.
    Event continues while (intensity >= I_seuil) OR (sum rain over next pdt_fin >= H_fin).
    Events are kept if total event rain >= H_min.
    """
    rain_mm = rain_mm.fillna(0.0)
    idx = rain_mm.index

    # intensity (mm/h) from mm/pdt
    intensity = rain_mm * 60.0 / pdt_min

    # future window sum from t to t + (steps-1)
    steps = int(round(pdt_fin_min / pdt_min))
    future_sum = rain_mm.rolling(window=steps, min_periods=1).sum().shift(-(steps - 1))

    # CEREMA "continue" condition (rain-only)
    active = (intensity >= I_seuil_mm_h) | (future_sum >= H_fin_mm)
    active = active.fillna(False).to_numpy()

    events = []
    i = 0
    n = len(active)

    while i < n:
        if not active[i]:
            i += 1
            continue

        # CEREMA uses ligne-1 (one step earlier) for start timestamp
        start_idx = max(i - 1, 0)
        start_time = idx[start_idx]

        j = i
        while j < n and active[j]:
            j += 1

        end_time = idx[j - 1]
        total = float(rain_mm.iloc[i:j].sum())

        if total >= H_min_mm:
            events.append((start_time, end_time, total))

        i = j

    return pd.DataFrame(events, columns=["date_start", "date_finish", "rainfall_total_mm"])


def merge_event_intervals(events: pd.DataFrame, gap_minutes: int = 0) -> pd.DataFrame:
    """
    Merge overlapping (or near-overlapping) [start, end] intervals.
    gap_minutes=0 merges only overlaps/touching; increase to merge small gaps.
    """
    if events.empty:
        return events.copy()

    gap = pd.Timedelta(minutes=gap_minutes)
    ev = events.sort_values("date_start").reset_index(drop=True)

    merged = []
    cur_s = ev.loc[0, "date_start"]
    cur_e = ev.loc[0, "date_finish"]
    cur_r = ev.loc[0, "rainfall_total_mm"]

    for k in range(1, len(ev)):
        s = ev.loc[k, "date_start"]
        e = ev.loc[k, "date_finish"]
        r = ev.loc[k, "rainfall_total_mm"]

        if s <= (cur_e + gap):  # overlap or small gap
            cur_e = max(cur_e, e)
            cur_r += r
        else:
            merged.append((cur_s, cur_e, cur_r))
            cur_s, cur_e, cur_r = s, e, r

    merged.append((cur_s, cur_e, cur_r))
    return pd.DataFrame(merged, columns=["date_start", "date_finish", "rainfall_total_mm"])

def build_rain_mask(index: pd.DatetimeIndex, events: pd.DataFrame, post_rain_delay_min: int = 150) -> pd.Series:
    """
    Create a boolean mask (True/False) for each timestamp in 'index':
    True if the timestamp is inside any [date_start, date_finish + post_delay].
    """
    mask = pd.Series(False, index=index)

    delay = pd.Timedelta(minutes=post_rain_delay_min)
    for _, r in events.iterrows():
        start = r["date_start"]
        end = r["date_finish"] + delay
        mask.loc[(mask.index >= start) & (mask.index <= end)] = True

    return mask
    
# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if not MAIN_TIMESERIES_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {MAIN_TIMESERIES_FILE}. Put your raw Cerema file in Inputs/ and update MAIN_TIMESERIES_FILE."
        )

    # Load raw data
    df = load_main_timeseries(MAIN_TIMESERIES_FILE)

    # --- Rain events using BOTH gauges (CEREMA rules) ---
    events_p1 = detect_rain_events_cerema(df["pluvio1"])
    events_p7 = detect_rain_events_cerema(df["pluvio7"])

    # Union of intervals from both gauges
    events_all = pd.concat([events_p1, events_p7], ignore_index=True)
    events_all = merge_event_intervals(events_all, gap_minutes=0)

    # Save event tables (Outputs/ is local; typically ignored by git)
    events_p1.to_csv(OUTPUTS_DIR / "rain_events_pluvio1.csv", index=False)
    events_p7.to_csv(OUTPUTS_DIR / "rain_events_pluvio7.csv", index=False)
    events_all.to_csv(OUTPUTS_DIR / "rain_events_both_gauges.csv", index=False)

    # --- Rain mask with post-rain delay (minutes) ---
    POST_RAIN_DELAY_MIN = 150  # change to 120 if you want later

    rain_mask = build_rain_mask(df.index, events_all, post_rain_delay_min=POST_RAIN_DELAY_MIN)
    # Save mask (useful for QA/QC)
    rain_mask.to_frame(name="rain_affected").to_csv(OUTPUTS_DIR / "rain_mask_with_delay.csv")
    
    # Resample to 30 minutes
    df_30 = resample_30min(df)

    # Save output
    out_csv = OUTPUTS_DIR / "timeseries_30min.csv"
    df_30.to_csv(out_csv)

    print("Done ✅")
    print(f"Saved: {out_csv}")
    print("Saved rain event tables:")
    print(f" - {OUTPUTS_DIR / 'rain_events_pluvio1.csv'}")
    print(f" - {OUTPUTS_DIR / 'rain_events_pluvio7.csv'}")
    print(f" - {OUTPUTS_DIR / 'rain_events_both_gauges.csv'}")


if __name__ == "__main__":
    main()
