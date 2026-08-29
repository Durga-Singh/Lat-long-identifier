"""
find_nearby_from_latlong.py

Use this when your file ALREADY has latitude/longitude for some or all rows
and you just want to find every pair of stores within DISTANCE_THRESHOLD_KM
of each other. No API calls, no geocoding — just fast distance checking.

USAGE
    1. pip install pandas openpyxl scikit-learn   (usually already installed)
    2. Set INPUT_FILE, LAT_COLUMN, LON_COLUMN below to match your file.
    3. python find_nearby_from_latlong.py

Rows with a missing/blank/invalid lat or long are skipped automatically and
listed at the end so you know which ones couldn't be checked.
"""

import math

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ======================= CONFIG — edit these =======================

INPUT_FILE = "dummy_store_locations_raw.xlsx"                # your file (.xlsx or .csv)
SHEET_NAME = 0                             # sheet name/index if .xlsx, ignored for .csv

LAT_COLUMN = "Latitude"                    # change if your column is named differently
LON_COLUMN = "Longitude"                   # e.g. "Lat", "lat", "Y", etc.

ID_COLUMN = "Store code"                   # used to label pairs in the output
NAME_COLUMN = "Business name"              # used to label pairs in the output

DISTANCE_THRESHOLD_KM = 1.0

NEARBY_PAIRS_OUTPUT_FILE = "nearby_pairs_within_1km.xlsx"
SKIPPED_ROWS_OUTPUT_FILE = "rows_missing_latlong.xlsx"

# =====================================================================


def load_input(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_excel(path, sheet_name=SHEET_NAME, dtype=str, keep_default_na=False)


def main():
    print(f"Reading {INPUT_FILE} ...")
    df = load_input(INPUT_FILE)

    for col in (LAT_COLUMN, LON_COLUMN):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in the file. "
                              f"Available columns: {list(df.columns)}")

    df["_lat"] = pd.to_numeric(df[LAT_COLUMN], errors="coerce")
    df["_lon"] = pd.to_numeric(df[LON_COLUMN], errors="coerce")

    # basic sanity range check (catches swapped lat/long, stray text, etc.)
    valid_range = df["_lat"].between(-90, 90) & df["_lon"].between(-180, 180)

    geo = df[df["_lat"].notna() & df["_lon"].notna() & valid_range].reset_index(drop=True)
    skipped = df[~(df["_lat"].notna() & df["_lon"].notna() & valid_range)]

    print(f"{len(geo)} of {len(df)} rows have usable lat/long.")
    if len(skipped):
        skipped.drop(columns=["_lat", "_lon"]).to_excel(SKIPPED_ROWS_OUTPUT_FILE, index=False)
        print(f"{len(skipped)} rows skipped (missing/invalid lat or long) — "
              f"saved to {SKIPPED_ROWS_OUTPUT_FILE}")

    if len(geo) < 2:
        print("Not enough rows with valid lat/long to compare. Stopping.")
        return

    coords_rad = np.radians(geo[["_lat", "_lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = DISTANCE_THRESHOLD_KM / 6371.0088  # km -> radians for this Earth radius

    pairs = []
    seen = set()
    for i in range(len(geo)):
        idxs, dists = tree.query_radius(coords_rad[i:i + 1], r=radius_rad, return_distance=True)
        for j, d_rad in zip(idxs[0], dists[0]):
            if j == i:
                continue
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            dist_km = d_rad * 6371.0088
            pairs.append({
                f"{ID_COLUMN} A": geo.at[i, ID_COLUMN] if ID_COLUMN in geo.columns else i,
                f"{NAME_COLUMN} A": geo.at[i, NAME_COLUMN] if NAME_COLUMN in geo.columns else "",
                "Latitude A": geo.at[i, "_lat"],
                "Longitude A": geo.at[i, "_lon"],
                f"{ID_COLUMN} B": geo.at[j, ID_COLUMN] if ID_COLUMN in geo.columns else j,
                f"{NAME_COLUMN} B": geo.at[j, NAME_COLUMN] if NAME_COLUMN in geo.columns else "",
                "Latitude B": geo.at[j, "_lat"],
                "Longitude B": geo.at[j, "_lon"],
                "Distance_km": round(dist_km, 4),
            })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        pairs_df = pairs_df.sort_values("Distance_km")
    pairs_df.to_excel(NEARBY_PAIRS_OUTPUT_FILE, index=False)

    print(f"Found {len(pairs_df)} pairs within {DISTANCE_THRESHOLD_KM} km.")
    print(f"Saved to {NEARBY_PAIRS_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
