import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union, nearest_points

# Files
input_csv = "./Data/earthquakeq_judge copy.csv"
coastline_file = "coastline_25_50N_90_65W.geojson"
output_csv = "./Data/earthquakeq_judge.csv"

# Read the CSV
df = pd.read_csv(input_csv)

# Ensure the new column exists
df["coast delta"] = pd.NA

# Load coastline and project it to a CRS with meter units
coastline = gpd.read_file(coastline_file).to_crs("EPSG:5070")
coast_geometry = unary_union(coastline.geometry)

# Make sure latitude and longitude are numeric
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# Only process rows with both coordinates
valid = (
    df["latitude"].notna()
    & df["longitude"].notna()
)

def distance_to_coast(row):
    point = Point(row["longitude"], row["latitude"])

    # Convert point from longitude/latitude to projected meter coordinates
    point_projected = gpd.GeoSeries(
        [point],
        crs="EPSG:4326"
    ).to_crs("EPSG:5070").iloc[0]

    # Find nearest point on the coastline
    nearest = nearest_points(point_projected, coast_geometry)[1]

    # Distance is in meters because EPSG:5070 uses meters
    return point_projected.distance(nearest) / 1000

df.loc[valid, "coast delta"] = df.loc[valid].apply(
    distance_to_coast,
    axis=1
)

# Save the updated table
df.to_csv(output_csv, index=False)

print(f"Saved {output_csv}")
