# Internship Hydrology Workflow

Python workflow developed during my internship to clean, resample and analyse rainfall/flow/storage time series, estimate dry-weather baseflow from storage-tank filling behaviour, and separate rainfall-induced flow.

## What this project does
- Loads and cleans rainfall + flow + water level + stored volume data
- Marks rain vs dry periods using rain-event tables
- Detects storage filling segments from stored-volume changes
- Estimates baseflow from dV/dt and builds a continuous baseflow series
- Computes rainfall-induced flow = total flow - baseflow
- Produces diagnostic plots and monthly summaries

## Data
Raw input data are not included in this repository.
Place your files in `Inputs/` (see folder structure below).

## Folder structure
