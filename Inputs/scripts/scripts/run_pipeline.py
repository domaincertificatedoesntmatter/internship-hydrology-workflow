from pathlib import Path
import pandas as pd
import numpy as np


INPUTS_DIR = Path("Inputs")
OUTPUTS_DIR = Path("Outputs")

MAIN_TIMESERIES_FILE = INPUTS_DIR / "data_v2.txt"          
WATER_LEVEL_FILE     = INPUTS_DIR / "water_level_P115.txt" 

RESAMPLE_FREQ = "30min"  


def load_main_timeseries(path: Path) -> pd.DataFrame:
    """
    Reads the main Cerema time series file with columns:
    time; pluvio1; pluvio7; flowrate; waterlevel; storedvolume
    """
    df = pd.read_csv(
        path,
        skiprows=6,
        delimiter=";",
        header=None,
        names=["time", "pluvio1", "pluvio7", "flowrate", "waterlevel", "storedvolume"],
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("time").sort_index()

    # Replace known missing codes with NaN
    df.replace([-6999, -7999, -9999, -9.999], np.nan, inplace=True)

    # Make sensor values positive if they appear negative
    for c in ["flowrate", "waterlevel", "storedvolume"]:
        df[c] = df[c].abs()

    return df


def load_water_level(path: Path) -> pd.DataFrame:
    """
    Reads the P115 water level file with columns: datetime; waterlevel
    """
    pm = pd.read_csv(path, delimiter=";", skiprows=7, header=None, names=["datetime", "waterlevel"])
    pm["datetime"] = pd.to_datetime(pm["datetime"], format="%Y-%m-%d %H:%M:%S")
    pm = pm.set_index("datetime").sort_index()

    pm["waterlevel"] = pm["waterlevel"].replace([-6999, -7999, -9999, -9.999], np.nan)

    return pm


def resample_30min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to 30 min (rainfall summed, sensors averaged)
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


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if not MAIN_TIMESERIES_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {MAIN_TIMESERIES_FILE}. Put your input file inside Inputs/ and update the filename in the script."
        )

    df = load_main_timeseries(MAIN_TIMESERIES_FILE)
    df_30 = resample_30min(df)

    # Save a simple output so we confirm the pipeline runs
    out_csv = OUTPUTS_DIR / "timeseries_30min.csv"
    df_30.to_csv(out_csv)

    print("✅ Done")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
