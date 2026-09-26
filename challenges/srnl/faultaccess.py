import io
import pathlib
import zipfile

import pandas as pd
import geopandas as gpd
import requests

from bs4 import BeautifulSoup
from shapely.geometry import Point


# ==================================================
# Configuration
# ==================================================

INPUT_CSV = "./Data/earthquakeq_train.csv"
OUTPUT_CSV = "locations_with_fault_delta.csv"

LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

# Distance used to turn fault lines into fault-area polygons.
# The resulting area extends approximately this distance
# on either side of each mapped fault.
BUFFER_KILOMETERS = 5

USGS_FAULT_PAGE = (
    "https://www.usgs.gov/programs/earthquake-hazards/faults"
)

DOWNLOAD_DIRECTORY = pathlib.Path("usgs_fault_data")


# ==================================================
# Download the USGS fault data
# ==================================================

print("Opening the USGS fault database page...")

page_response = requests.get(
    USGS_FAULT_PAGE,
    timeout=60,
)

page_response.raise_for_status()

soup = BeautifulSoup(
    page_response.text,
    "html.parser",
)

# Find ZIP-file links on the USGS page.
zip_links = []

for link in soup.find_all("a", href=True):
    href = link["href"]

    if ".zip" not in href.lower():
        continue

    if href.startswith("/"):
        href = "https://www.usgs.gov" + href
    elif href.startswith("//"):
        href = "https:" + href

    zip_links.append(href)

if not zip_links:
    raise RuntimeError(
        "Could not find a ZIP download link on the USGS fault page."
    )

# Prefer a link that appears to be a GIS download.
gis_links = [
    link for link in zip_links
    if "gis" in link.lower()
]

usgs_zip_url = (
    gis_links[0]
    if gis_links
    else zip_links[0]
)

print(f"Downloading fault data from:\n{usgs_zip_url}")

zip_response = requests.get(
    usgs_zip_url,
    timeout=180,
)

zip_response.raise_for_status()

DOWNLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
    archive.extractall(DOWNLOAD_DIRECTORY)

print("USGS data downloaded and extracted.")


# ==================================================
# Locate the downloaded shapefile
# ==================================================

shapefiles = list(
    DOWNLOAD_DIRECTORY.rglob("*.shp")
)

if not shapefiles:
    raise FileNotFoundError(
        "No shapefiles were found in the downloaded USGS archive."
    )

print("\nShapefiles found:")

for number, shapefile in enumerate(shapefiles):
    print(f"{number}: {shapefile}")

# Use the first shapefile by default.
# If several are listed, change this index if necessary.
FAULT_SHAPEFILE = shapefiles[0]

print(f"\nUsing: {FAULT_SHAPEFILE}")

faults = gpd.read_file(FAULT_SHAPEFILE)

if faults.empty:
    raise RuntimeError("The fault shapefile contains no features.")

if faults.crs is None:
    raise ValueError(
        "The fault shapefile does not contain a coordinate reference system."
    )


# ==================================================
# Convert fault lines into fault-area polygons
# ==================================================

# EPSG:5070 uses meters and is appropriate for the
# contiguous United States.
faults_meters = faults.to_crs("EPSG:5070")

buffer_meters = BUFFER_KILOMETERS * 1000

# Buffer each fault line to create an area.
fault_areas_meters = faults_meters.copy()

fault_areas_meters.geometry = (
    fault_areas_meters.geometry.buffer(buffer_meters)
)

# Combine all fault areas into one geometry for distance calculations.
try:
    all_fault_areas = fault_areas_meters.geometry.union_all()
except AttributeError:
    # Compatibility with older Shapely versions
    all_fault_areas = fault_areas_meters.geometry.unary_union

print(
    f"Loaded {len(faults)} fault features."
)
print(
    f"Using a {BUFFER_KILOMETERS}-kilometer buffer around each fault."
)


# ==================================================
# Read the input CSV
# ==================================================

df = pd.read_csv(INPUT_CSV)

if LATITUDE_COLUMN not in df.columns:
    raise ValueError(
        f"Missing required column: {LATITUDE_COLUMN}"
    )

if LONGITUDE_COLUMN not in df.columns:
    raise ValueError(
        f"Missing required column: {LONGITUDE_COLUMN}"
    )

# Convert coordinate columns to numeric values.
# Invalid values and blanks become NaN.
df[LATITUDE_COLUMN] = pd.to_numeric(
    df[LATITUDE_COLUMN],
    errors="coerce",
)

df[LONGITUDE_COLUMN] = pd.to_numeric(
    df[LONGITUDE_COLUMN],
    errors="coerce",
)

# Create the output column.
df["fault delta"] = pd.NA

# Keep only rows with valid geographic coordinates.
valid_rows = (
    df[LATITUDE_COLUMN].notna()
    & df[LONGITUDE_COLUMN].notna()
    & df[LATITUDE_COLUMN].between(-90, 90)
    & df[LONGITUDE_COLUMN].between(-180, 180)
)

print(
    f"Rows with valid coordinates: {valid_rows.sum()} "
    f"of {len(df)}"
)


# ==================================================
# Calculate distance from each point to a fault area
# ==================================================

if valid_rows.any():

    points = gpd.GeoDataFrame(
        df.loc[valid_rows].copy(),
        geometry=[
            Point(longitude, latitude)
            for latitude, longitude in zip(
                df.loc[valid_rows, LATITUDE_COLUMN],
                df.loc[valid_rows, LONGITUDE_COLUMN],
            )
        ],
        crs="EPSG:4326",
    )

    # Reproject points into meters.
    points_meters = points.to_crs("EPSG:5070")

    # Distance is zero for points inside a fault area.
    # Otherwise, it is the distance to the nearest fault-area edge.
    distances_meters = points_meters.geometry.distance(
        all_fault_areas
    )

    distances_kilometers = (
        distances_meters / 1000
    ).round(3)

    # Write distances back to the original DataFrame.
    df.loc[valid_rows, "fault delta"] = (
        distances_kilometers.to_numpy()
    )


# ==================================================
# Write the new CSV
# ==================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(f"\nFinished.")
print(f"Output written to: {OUTPUT_CSV}")
