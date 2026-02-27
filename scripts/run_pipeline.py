from pathlib import Path
import numpy as np
import pandas as pd

INPUTS_DIR = Path("Inputs")
OUTPUTS_DIR = Path("Outputs")

# Rename these to match your real input file names (we will do that later)
MAIN_TIMESERIES_FILE = INPUTS_DIR / "data_v2.txt"
WATER_LEVEL_FILE = INPUTS_DIR / "water_level_P115.txt"

RESAMPLE_FREQ = "30min"


def load_main_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=6,
        delimiter=";",
        header=None,
        names=["time", "pluvio1", "pluvio7", "flowrate", "waterlevel", "storedvolume"],
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("time").sort_index()

    # Replace known missing-value codes with NaN
    df.replace([-6999, -7999, -9999, -9.999], np.nan, inplace=True)

    # Some sensors can appear negative; take absolute values
    for c in ["flowrate", "waterlevel", "storedvolume"]:
        df[c] = df[c].abs()

    return df


def resample_30min(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample(RESAMPLE_FREQ).agg(
        {
            "pluvio1": "sum",
            "pluvio7": "sum",
            "flowrate": "mean",
            "waterlevel": "mean",
            "storedvolume": "mean",
        }
    )


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if not MAIN_TIMESERIES_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {MAIN_TIMESERIES_FILE}. Put your file in Inputs/ and update MAIN_TIMESERIES_FILE."
        )

    df = load_main_timeseries(MAIN_TIMESERIES_FILE)
    df_30 = resample_30min(df)

    out_csv = OUTPUTS_DIR / "timeseries_30min.csv"
    df_30.to_csv(out_csv)

    print("Done ✅")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
