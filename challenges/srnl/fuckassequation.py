# CHAT: Document the corrected units, the paper reference, and compatibility with the CSV batch script.
"""ETAS features using Chu et al. (2011), Equations (3)-(4), Table 2 Zone 0.

Single row:
    python fuckassequation.py Data/earthquakeq_train.csv 73
Batch using the previously supplied companion script:
    python add_etas_to_csv.py Data/earthquakeq_train.csv Data/earthquakeq_train_with_etas.csv --equations fuckassequation.py

# CHAT: Add magnitude-sum feature examples; this feature does not modify the ETAS equations.
Single-row historical magnitude sums:
    python fuckassequation.py Data/earthquakeq_train.csv 73 --charlesonian
Export the four bands for every seismic row:
    python fuckassequation.py Data/earthquakeq_train.csv --charlesonian-output Data/earthquakeq_train_charlesonian.csv
Freeze training history when scoring another file:
    python fuckassequation.py Data/earthquakeq_test.csv --charlesonian-output Data/earthquakeq_test_charlesonian.csv --history-csv Data/earthquakeq_train.csv
# CHAT: The default Charlesonian window is now 1,825 days; no magnitude cutoff is applied.
Use --history-days to override it, or --history-mag-min 1.75 for a strict
magnitude cutoff. All magnitudes are included by default. Current-time and future events never contribute.

CHANGES (search for '# CHAT:' for descriptions beside edited code):
* Exact continuous time integral instead of the 30-sample daily sum.
* Adaptive spherical cell integration for triggering, with a convergence check.
* Consistent count units for background plus triggering; one Earth radius.
* Separate instantaneous intensity and frozen-history probability approximation.
* Keep magnitude reference 5.0 (paper) separate from parent cutoff 1.75 (user).
* Expose forecast length, accuracy, and parameter overrides; cache spatial work.

The published constants are not calibrated to this local M>1.75 catalog. They
were fitted to shallow M>=5 earthquakes in plate interiors. The calculation
retains the user's strict 180-day/200-km parent filters, applies no depth/zone
filter, and excludes simultaneous/future events. Naive timestamps mean UTC.
The approximate score excludes further triggering by events that occur during
the forecast. It is a model feature, not a validated 30-day hazard probability.
The helper functions and return keys used by add_etas_to_csv.py remain available.
"""

import argparse
import csv
import json
import os  # CHAT: Atomically publish completed Charlesonian CSV exports.
import tempfile  # CHAT: Keep partial CSV exports from replacing a complete file.
from pathlib import Path  # CHAT: Resolve and protect separate history/target CSV paths.
from datetime import datetime as DateTime, timezone
from functools import lru_cache  # CHAT: Reuse cell nodes and spatial integrals across CSV rows.
import math as m
import numpy as np

# CHAT: Use the same spherical Earth radius for distances and cell areas.
EARTH_RADIUS_KM = 6371.0088

class s_event:
    lat: float #deg
    lng: float #deg
    mag: float
    day: float # fractional UTC days since 1970-01-01
    def __init__(self, lat, lng, mag, day):
        self.lat = lat
        self.lng = lng
        self.mag = mag
        self.day = day




import math

def distance_km(lat1, lon1, lat2, lon2):
    earth_radius = EARTH_RADIUS_KM  # CHAT: Match the radius used for spherical cell areas.

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c


import numpy as np

def make_latlon_grid(
    lat_min=25.0,
    lat_max=50.0,
    lon_min=-90.0,
    lon_max=-65.0,
    cell_size=0.5,
    earth_radius_km=6371.0088,
):
    """
    Create a latitude/longitude grid and calculate the spherical area
    of every cell.

    Longitudes use the degrees-east convention:
        65° W = -65
        90° W = -90

    Returns
    -------
    cells : numpy structured array
        Fields:
        lat_min, lat_max, lon_min, lon_max,
        lat_center, lon_center, area_km2
    """

    # Create cell edges
    lat_edges = np.arange(lat_min, lat_max + cell_size, cell_size)
    lon_edges = np.arange(lon_min, lon_max + cell_size, cell_size)

    # Ensure exact final boundaries
    lat_edges[-1] = lat_max
    lon_edges[-1] = lon_max

    # Cell boundary arrays
    lat_south, lon_west = np.meshgrid(
        lat_edges[:-1],
        lon_edges[:-1],
        indexing="ij"
    )

    lat_north, lon_east = np.meshgrid(
        lat_edges[1:],
        lon_edges[1:],
        indexing="ij"
    )

    # Cell centers
    lat_center = (lat_south + lat_north) / 2
    lon_center = (lon_west + lon_east) / 2

    # Convert angular distances to radians
    lat_south_rad = np.radians(lat_south)
    lat_north_rad = np.radians(lat_north)
    dlon_rad = np.radians(lon_east - lon_west)

    # Spherical quadrilateral area:
    # A = R² * Δlongitude * (sin(latitude_north) - sin(latitude_south))
    area_km2 = (
        earth_radius_km**2
        * dlon_rad
        * (
            np.sin(lat_north_rad)
            - np.sin(lat_south_rad)
        )
    )

    # Store all cells as a one-dimensional structured array
    n_cells = area_km2.size

    cells = np.empty(
        n_cells,
        dtype=[
            ("lat_min", "f8"),
            ("lat_max", "f8"),
            ("lon_min", "f8"),
            ("lon_max", "f8"),
            ("lat_center", "f8"),
            ("lon_center", "f8"),
            ("area_km2", "f8"),
        ],
    )

    cells["lat_min"] = lat_south.ravel()
    cells["lat_max"] = lat_north.ravel()
    cells["lon_min"] = lon_west.ravel()
    cells["lon_max"] = lon_east.ravel()
    cells["lat_center"] = lat_center.ravel()
    cells["lon_center"] = lon_center.ravel()
    cells["area_km2"] = area_km2.ravel()

    return cells

def get_cell(
    lat,
    lon,
    lat_min=25.0,
    lat_max=50.0,
    lon_min=-90.0,
    lon_max=-65.0,
    cell_size=0.5,
):
    """
    Return the grid cell containing a latitude/longitude point.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees east.
        For example, 90° W = -90.

    Returns
    -------
    dict
        Cell boundaries, center, and zero-based row/column indices.

    Raises
    ------
    ValueError
        If the point lies outside the grid.
    """

    # Check whether the point is inside the grid
    if not (lat_min <= lat <= lat_max):
        raise ValueError(f"Latitude {lat} is outside the grid.")

    if not (lon_min <= lon <= lon_max):
        raise ValueError(f"Longitude {lon} is outside the grid.")

    # Calculate zero-based cell indices
    row = int((lat - lat_min) // cell_size)
    col = int((lon - lon_min) // cell_size)

    # Handle points exactly on the maximum boundary
    n_rows = int(round((lat_max - lat_min) / cell_size))
    n_cols = int(round((lon_max - lon_min) / cell_size))

    row = min(row, n_rows - 1)
    col = min(col, n_cols - 1)

    # Calculate cell boundaries
    cell_lat_min = lat_min + row * cell_size
    cell_lat_max = cell_lat_min + cell_size

    cell_lon_min = lon_min + col * cell_size
    cell_lon_max = cell_lon_min + cell_size

    return {
        "row": row,
        "col": col,
        "lat_min": cell_lat_min,
        "lat_max": cell_lat_max,
        "lon_min": cell_lon_min,
        "lon_max": cell_lon_max,
        "lat_center": (cell_lat_min + cell_lat_max) / 2,
        "lon_center": (cell_lon_min + cell_lon_max) / 2,
    }

import numpy as np

def generate_random_events(
    n_events,
    lat_min=25.0,
    lat_max=50.0,
    lon_min=-90.0,
    lon_max=-65.0,
    mag_min=1.0,
    mag_max=5.0,
    day_min=0,
    day_max=180,
    seed=None,
):
    """Generate random seismic events within a geographic region."""

    rng = np.random.default_rng(seed)

    events = []

    for _ in range(n_events):
        event = s_event(
            lat=float(rng.uniform(lat_min, lat_max)),
            lng=float(rng.uniform(lon_min, lon_max)),
            mag=float(rng.uniform(mag_min, mag_max)),
            day=int(rng.integers(day_min, day_max + 1)),
        )

        events.append(event)

    return events


# CHAT: Centralize the unchanged Table 2, Zone 0 constants so fitted values can be supplied explicitly.
PAPER_ZONE0_PARAMETERS = {
    "K": 0.146, "a": 0.406, "c": 0.566, "p": 1.22,
    "d": 60.5, "q": 1.52, "mu": 0.134e-9,
}


# CHAT: Replace the 30-point daily sum with the analytic continuous-time integral.
def _time_integral(dt, c, p, forecast_days):
    """Integral of (dt + s + c)^(-p) for s in [0, forecast_days]."""
    lower = dt + c
    log_ratio = m.log1p(forecast_days / lower)
    if p == 1.0:
        return log_ratio  # CHAT: Handle p=1 without division by zero.
    exponent = 1.0 - p
    # CHAT: expm1/log1p avoid cancellation for short horizons or p near one.
    return lower ** exponent * m.expm1(exponent * log_ratio) / exponent


# CHAT: Cache spherical quadrature nodes for repeated predictions in the same cell.
@lru_cache(maxsize=64)
def _cell_quadrature(lat_min, lat_max, lon_min, lon_max, order):
    """Gauss-Legendre nodes in sin(latitude), longitude; weights are km^2."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    sin_lo, sin_hi = m.sin(m.radians(lat_min)), m.sin(m.radians(lat_max))
    lon_lo, lon_hi = m.radians(lon_min), m.radians(lon_max)
    sin_lat = 0.5 * (sin_hi + sin_lo) + 0.5 * (sin_hi - sin_lo) * nodes
    lon = 0.5 * (lon_hi + lon_lo) + 0.5 * (lon_hi - lon_lo) * nodes
    lat_grid, lon_grid = np.meshgrid(np.arcsin(sin_lat), lon, indexing="ij")
    # CHAT: dA = R^2 * d(sin(latitude)) * d(longitude), avoiding planar-area approximations.
    area_weights = (
        EARTH_RADIUS_KM ** 2 * (sin_hi - sin_lo) * (lon_hi - lon_lo) / 4.0
        * np.outer(weights, weights)
    )
    arrays = (lat_grid.ravel(), lon_grid.ravel(), area_weights.ravel())
    for array in arrays:
        array.setflags(write=False)
    return arrays


# CHAT: Integrate each earthquake's spatial density across the cell, instead of using one point.
@lru_cache(maxsize=8192)
def _spatial_integral(event_lat, event_lng, magnitude, bounds,
                      a, d, q, magnitude_reference, rtol, max_order):
    """Return (spatial integral, successive-order difference, final order).

    Doubling quadrature order estimates convergence; the difference is an
    error estimate, not a rigorous bound. Failure to converge raises an error.
    Cache keys contain every spatial input, so time-independent integrals can
    be reused safely by the batch CSV script.
    """
    source_lat, source_lng = m.radians(event_lat), m.radians(event_lng)
    magnitude_scale = m.exp(-a * (magnitude - magnitude_reference))
    previous = None
    order = 16
    while order <= max_order:
        lat, lng, weights = _cell_quadrature(*bounds, order)
        hav = (
            np.sin((lat - source_lat) / 2.0) ** 2
            + m.cos(source_lat) * np.cos(lat) * np.sin((lng - source_lng) / 2.0) ** 2
        )
        # CHAT: Use great-circle distances at every quadrature node with roundoff protection.
        distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
        kernel = (distance ** 2 * magnitude_scale + d) ** (-q)
        integral = float(np.dot(weights, kernel))
        if previous is not None:
            error = abs(integral - previous)
            if error <= max(1e-14, rtol * abs(integral)):
                return integral, error, order
        previous = integral
        order *= 2
    raise ArithmeticError(
        f"Spatial integral did not converge at order {max_order} for "
        f"earthquake ({event_lat}, {event_lng}). Increase quadrature_max_order."
    )


# CHAT: Rewrite the core to keep spatial density, cell counts, and the probability approximation distinct.
def _calculate_etas(
    events, lat, lng, current_day, dt_max=180.0,
    dist_max=200.0, mag_min=1.75, cell_size=0.5, *,
    forecast_days=30.0, magnitude_reference=5.0,
    quadrature_rtol=1e-6, quadrature_max_order=128, parameters=None,
):
    """Compute an instantaneous point intensity and a finite-cell forecast feature.

    etas_score / probability / probability_approx:
        1-exp(-total_lambda), a frozen-history Poisson approximation, NOT a
        calibrated full ETAS probability. New events during the forecast do
        not become new parents in this calculation.
    event_lambda / background_lambda / total_lambda:
        Integrated expected counts from direct offspring of selected existing
        parents, background immigrants, and their sum, respectively. Further
        offspring of events occurring during the forecast are not included.
    etas_intensity:
        Equation (3) evaluated at the target time/location in events/day/km^2,
        with the user-selected history cutoffs.

    Selection preserves 0 < dt < dt_max, distance TO TARGET < dist_max,
    and parent magnitude > mag_min. Once selected, each parent's kernel is
    integrated over the WHOLE cell without an additional node-distance cutoff.
    These history cutoffs are application choices, not fitted paper parameters.

    The unchanged paper constants were fitted to shallow M>=5 plate-interior
    events. magnitude_reference=5.0 keeps that reference in Equation (4).
    mag_min=1.75 is retained as the user's parent-selection threshold; including
    smaller events is an uncalibrated extrapolation, not a refit for M>1.75.
    Depth and tectonic zone are not filtered by this function. To reproduce the
    paper's population, prefilter the catalog appropriately and use its cutoff.
    """
    values = (lat, lng, current_day, dt_max, dist_max, mag_min, cell_size,
              forecast_days, magnitude_reference, quadrature_rtol)
    if not all(m.isfinite(x) for x in values):
        raise ValueError("Coordinates, time, thresholds, and integration settings must be finite.")
    if min(dt_max, dist_max, cell_size, forecast_days) <= 0:
        raise ValueError("Time windows, dist_max, and cell_size must be positive.")
    if not 0 < quadrature_rtol < 1:
        raise ValueError("quadrature_rtol must be between 0 and 1.")
    if (isinstance(quadrature_max_order, bool)
            or not isinstance(quadrature_max_order, (int, np.integer))
            or quadrature_max_order < 32
            or quadrature_max_order > 512
            or quadrature_max_order & (quadrature_max_order - 1)):
        raise ValueError("quadrature_max_order must be a power of two from 32 to 512.")
    count = 25.0 / cell_size
    if count < 1 or not m.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("cell_size must divide the original 25-degree grid span.")

    # CHAT: Allow explicit parameter overrides after fitting, without silently altering the published constants.
    coefficients = PAPER_ZONE0_PARAMETERS.copy()
    if parameters is not None:
        unknown = set(parameters) - set(coefficients)
        if unknown:
            raise ValueError(f"Unknown ETAS parameters: {sorted(unknown)}")
        coefficients.update(parameters)
    if not all(m.isfinite(v) for v in coefficients.values()):
        raise ValueError("ETAS parameters must be finite.")
    if (min(coefficients[k] for k in ("K", "a", "mu")) < 0
            or min(coefficients[k] for k in ("c", "d", "p", "q")) <= 0):
        raise ValueError("Require K,a,mu >= 0 and c,d,p,q > 0.")
    K, a, c, p, d, q, mu = (coefficients[k] for k in ("K", "a", "c", "p", "d", "q", "mu"))
    cell = get_cell(lat, lng, cell_size=cell_size)
    bounds = tuple(cell[k] for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
    cell_area_km2 = (
        EARTH_RADIUS_KM ** 2 * m.radians(cell["lon_max"] - cell["lon_min"])
        * (m.sin(m.radians(cell["lat_max"])) - m.sin(m.radians(cell["lat_min"])))
    )

    event_counts, point_rates, estimated_errors = [], [], []
    events_loaded = events_used = invalid_events = spatial_max_order = 0
    for event in events:
        events_loaded += 1
        # CHAT: Validate direct callers' event objects too; CSV parsing already rejects these values.
        if not (all(m.isfinite(v) for v in (event.lat, event.lng, event.mag, event.day))
                and -90 <= event.lat <= 90 and -180 <= event.lng <= 180):
            invalid_events += 1
            continue
        dt = current_day - event.day
        if not (0 < dt < dt_max and event.mag > mag_min):
            continue
        dist = distance_km(event.lat, event.lng, lat, lng)
        if dist >= dist_max:
            continue
        # CHAT: Keep paper reference magnitude separate from the eligibility threshold.
        dec = m.exp(-a * (event.mag - magnitude_reference))
        point_kernel = (dist ** 2 * dec + d) ** (-q)
        point_rates.append(K * (dt + c) ** (-p) * point_kernel)
        temporal_integral = _time_integral(dt, c, p, forecast_days)
        spatial_integral, error, order = _spatial_integral(
            event.lat, event.lng, event.mag, bounds,
            a, d, q, magnitude_reference, quadrature_rtol, quadrature_max_order,
        )
        event_counts.append(K * temporal_integral * spatial_integral)
        estimated_errors.append(K * temporal_integral * error)
        spatial_max_order = max(spatial_max_order, order)
        events_used += 1

    # CHAT: Both terms are now cell-wide counts over the same forecast interval.
    event_lambda = m.fsum(event_counts)
    background_lambda = mu * forecast_days * cell_area_km2
    total_lambda = background_lambda + event_lambda
    probability_approx = -m.expm1(-total_lambda)
    return {
        # CHAT: Retain the old keys for add_etas_to_csv.py while explicitly naming the approximation.
        "etas_score": probability_approx,
        "probability": probability_approx,
        "probability_approx": probability_approx,
        "event_lambda": event_lambda,
        "background_lambda": background_lambda,
        "total_lambda": total_lambda,
        "etas_intensity": mu + m.fsum(point_rates),
        "intensity_units": "events/day/km^2",
        "lambda_units": "events per cell over forecast_days; frozen history",
        "events_loaded": events_loaded,
        "events_used": events_used,
        "invalid_events": invalid_events,
        "cell_area_km2": cell_area_km2,
        "forecast_days": forecast_days,
        "magnitude_reference": magnitude_reference,
        "history_mag_min": mag_min,
        "parameters": coefficients,
        "spatial_max_order": spatial_max_order,
        "spatial_count_error_estimate": m.fsum(estimated_errors),
        "calibration_note": (
            "Default parameters: Chu et al. (2011), Table 2, Zone 0; shallow M>=5 events. "
            "Not fitted to this CSV or a lower magnitude threshold. No depth or zone filter applied. "
            "Probability approximation omits new triggering parents during the forecast."
        ),
    }


# CHAT: Preserve the existing scalar-return API and forward the new model options.
def calculate_probability(
    events, lat=40.0, lng=-70.0, current_day=180.0,
    dt_max=180.0, dist_max=200.0, mag_min=1.75, **model_options,
):
    """Return the frozen-history cell probability approximation."""
    return _calculate_etas(
        events, lat, lng, current_day, dt_max, dist_max, mag_min, **model_options
    )["probability_approx"]


def simulate_probabilities(
    n_simulations=1000,
    n_events=1000,
):
    probabilities = []

    for seed in range(n_simulations):
        events = generate_random_events(
            n_events=n_events,
            mag_min=1.75,
            seed=seed
        )

        probability = calculate_probability(events)
        probabilities.append(probability)

    return np.array(probabilities)

REQUIRED_COLUMNS = {"source", "time", "latitude", "longitude", "magnitude"}
_EPOCH = DateTime(1970, 1, 1, tzinfo=timezone.utc)


def _parse_time(value):
    """Read ISO timestamps, including fractional seconds and UTC offsets."""
    if isinstance(value, DateTime):
        result = value
    else:
        result = DateTime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _to_day(value):
    return (_parse_time(value) - _EPOCH).total_seconds() / 86400.0


def _coordinates(row):
    lat, lng = float(row["latitude"]), float(row["longitude"])
    if not (m.isfinite(lat) and m.isfinite(lng)
            and -90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("Invalid latitude or longitude.")
    return lat, lng


def _csv_rows(csv_filename):
    with open(csv_filename, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        yield from reader


def _is_seismic(row):
    return (row.get("source") or "").strip().lower() == "seismic"


def _iter_events_from_csv(csv_filename):
    """Stream valid seismic history; ignore other sources and malformed history.

    Depth and station/parameter/value fields do not enter the original equations.
    History may appear in any row order: eligibility uses timestamps.
    """
    for row in _csv_rows(csv_filename):
        if not _is_seismic(row):
            continue
        try:
            lat, lng = _coordinates(row)
            mag = float(row["magnitude"])
            day = _to_day(row["time"])
            if not m.isfinite(mag):
                continue
        except (TypeError, ValueError, AttributeError, OverflowError):
            continue
        yield s_event(lat, lng, mag, day)


def load_events_from_csv(csv_filename):
    """Load valid seismic events with fractional UTC days from the time column."""
    return list(_iter_events_from_csv(csv_filename))


# CHAT: Forward new integration/parameter options while retaining the old positional arguments.
def probability_from_csv(
    lat, lng, datetime, csv_filename, mag_min=1.75,
    dist_max=200.0, cell_size=0.5, dt_max=180.0,
    **model_options,  # CHAT: Optional forecast_days, magnitude_reference, quadrature settings, parameters.
):
    """Score an explicit location/time using strictly earlier seismic history.

    `datetime` accepts an ISO timestamp or datetime object. The original grid
    covers latitude 25..50 and longitude -90..-65; targets outside it raise
    ValueError. etas_score is a frozen-history cell probability approximation.
    etas_intensity is the instantaneous point rate in events/day/km^2.
    See _calculate_etas for the units and calibration limitations.
    """
    return _calculate_etas(
        _iter_events_from_csv(csv_filename), lat, lng, _to_day(datetime),
        dt_max=dt_max, dist_max=dist_max, mag_min=mag_min, cell_size=cell_size,
        **model_options,  # CHAT: Pass the requested numerical and model settings to the core.
    )


# CHAT: Keep zero-based row selection and allow the corrected calculation to be configured.
def etas_from_csv_row(
    csv_filename, row_index, *, dt_max=180.0, dist_max=200.0,
    mag_min=1.75, cell_size=0.5,
    **model_options,  # CHAT: Forward optional forecast and integration settings.
):
    """Calculate ETAS for ONE selected seismic row, not for the whole dataset.

    row_index is a zero-based data-row index across all sources, excluding the
    header. The target's time, latitude, and longitude come from that row;
    its magnitude is not used as a predictor. Non-seismic/invalid targets raise
    ValueError; missing indices raise IndexError. Invalid historical rows are
    skipped. events_loaded counts all valid seismic rows encountered in history;
    events_used counts only those meeting the time/distance/magnitude filters.

    CSV has no random-access index: reading the target scans up to row_index,
    then history is streamed once without assuming chronological file order.
    Only one ETAS score is computed. Events at or after the target timestamp
    (including the target itself) never contribute.
    """
    if isinstance(row_index, bool) or not isinstance(row_index, (int, np.integer)):
        raise TypeError("row_index must be a zero-based integer.")
    if row_index < 0:
        raise IndexError("row_index must be nonnegative.")
    rows = _csv_rows(csv_filename)
    try:
        for index, row in enumerate(rows):
            if index == row_index:
                target = row
                break
        else:
            raise IndexError(f"row_index {row_index} is outside the CSV data rows.")
    finally:
        rows.close()

    if not _is_seismic(target):
        raise ValueError(f"Row {row_index} is not a seismic row.")
    try:
        lat, lng = _coordinates(target)
        time = _parse_time(target["time"])
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Row {row_index} has invalid time or coordinates.") from exc
    result = probability_from_csv(
        lat, lng, time, csv_filename, mag_min=mag_min,
        dist_max=dist_max, cell_size=cell_size, dt_max=dt_max,
        **model_options,  # CHAT: Forward options without using the target magnitude as a predictor.
    )
    result.update({
        "row_index": int(row_index), "time": time.isoformat(),
        "latitude": lat, "longitude": lng, "current_day": _to_day(time),
    })
    return result


# CHAT: Add four disjoint distance bands for magnitude-weighted historical earthquake activity.
CHARLESONIAN_DISTANCE_EDGES_KM = (0.0, 25.0, 50.0, 100.0, 200.0)
# CHAT: Use a fixed five-year (1,825-day) history by default for comparable features.
CHARLESONIAN_LOOKBACK_DAYS = 5 * 365


# CHAT: Prepare a time-sorted, vectorized history once for efficient repeated point queries.
class _CharlesonianCatalog:
    def __init__(self, events):
        data = np.asarray([(e.day, e.lat, e.lng, e.mag) for e in events], dtype=float).reshape(-1, 4)
        valid = (np.isfinite(data).all(axis=1)
                 & (np.abs(data[:, 1]) <= 90) & (np.abs(data[:, 2]) <= 180))
        data = data[valid]
        data = data[np.argsort(data[:, 0], kind="stable")]
        self.days = data[:, 0]
        self.latitudes = np.radians(data[:, 1])
        self.longitudes = np.radians(data[:, 2])
        self.magnitudes = data[:, 3]


# CHAT: Validate configurable band edges and keep CSV column names tied to their actual distances.
def _charlesonian_columns(distance_edges_km):
    edges = np.asarray(distance_edges_km, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] != 0 or not np.all(np.diff(edges) > 0)):
        raise ValueError("Distance edges must be finite, strictly increasing, and start at zero.")
    return edges, [f"Charlesonian_{lo:g}_{hi:g}_km" for lo, hi in zip(edges[:-1], edges[1:])]


# CHAT: Add the requested TheCharlesonian function; weights are each historical event's own magnitude.
def TheCharlesonian(
    events, lat, lng, current_day, *,
    distance_edges_km=CHARLESONIAN_DISTANCE_EDGES_KM,
    lookback_days=CHARLESONIAN_LOOKBACK_DAYS, mag_min=None,
):
    """Return one magnitude sum per distance band for a point at a given time.

    Default bands are [0,25), [25,50), [50,100), [100,200] km. A historical
    earthquake contributes its OWN magnitude exactly once, not the target's
    magnitude. No weights are learned here. These are raw ML input features.

    events: s_event iterable (use load_events_from_csv for seismic-only input)
            or a prepared _CharlesonianCatalog for repeated queries.
    current_day: numeric day on the events' time origin, or an ISO timestamp/
                 datetime when the events use the CSV loader's UTC epoch.
    lookback_days: defaults to 1,825 days (five 365-day years). Explicit None
                   means all supplied earlier history; a positive value
                   restricts history to current_day-lookback_days < day < current_day.
    mag_min: None includes all valid magnitudes (including zero/negative ones);
             otherwise require event.mag > mag_min. Depth is not filtered.

    Early rows with less than five years of supplied history use a partial
    window. The combined add_etas_to_csv exporter flags these for exclusion
    when fitting weights; this point-query function does not drop observations.
    Events at or after current_day are always excluded, even in an unsorted
    catalog. Coordinates can be anywhere on Earth; no ETAS grid is required.
    An empty eligible history returns zeros, not missing values. These sums
    measure recorded activity, not released energy or geological stability.
    """
    if isinstance(current_day, (str, DateTime)):
        current_day = _to_day(current_day)
    if not (all(m.isfinite(v) for v in (lat, lng, current_day))
            and -90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("The target requires valid coordinates and a finite time.")
    if lookback_days is not None and (not m.isfinite(lookback_days) or lookback_days <= 0):
        raise ValueError("lookback_days must be positive and finite, or None.")
    if mag_min is not None and not m.isfinite(mag_min):
        raise ValueError("mag_min must be finite, or None.")
    edges, columns = _charlesonian_columns(distance_edges_km)
    catalog = events if isinstance(events, _CharlesonianCatalog) else _CharlesonianCatalog(events)
    # CHAT: Enforce prediction-time history, preventing self, simultaneous-event, and future-event leakage.
    stop = int(np.searchsorted(catalog.days, current_day, side="left"))
    start = 0 if lookback_days is None else int(
        np.searchsorted(catalog.days, current_day - lookback_days, side="right")
    )
    latitudes, longitudes = catalog.latitudes[start:stop], catalog.longitudes[start:stop]
    magnitudes = catalog.magnitudes[start:stop]
    if mag_min is not None:
        keep = magnitudes > mag_min
        latitudes, longitudes, magnitudes = latitudes[keep], longitudes[keep], magnitudes[keep]
    target_lat, target_lng = m.radians(lat), m.radians(lng)
    hav = (np.sin((latitudes - target_lat) / 2) ** 2
           + np.cos(latitudes) * m.cos(target_lat) * np.sin((longitudes - target_lng) / 2) ** 2)
    distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    # CHAT: Snap sub-micrometre distance roundoff at edges so an exact boundary is assigned consistently.
    for edge in edges:
        distances[np.isclose(distances, edge, rtol=0.0, atol=1e-9)] = edge
    result = {}
    for index, (lo, hi, column) in enumerate(zip(edges[:-1], edges[1:], columns)):
        upper = distances <= hi if index == len(columns) - 1 else distances < hi
        selected = magnitudes[(distances >= lo) & upper]
        result[column] = m.fsum(selected.tolist())
    return result


# CHAT: Provide the same zero-based CSV row workflow for TheCharlesonian as for ETAS.
def charlesonian_from_csv_row(
    csv_filename, row_index, *, history_csv=None,
    lookback_days=CHARLESONIAN_LOOKBACK_DAYS, mag_min=None,
    distance_edges_km=CHARLESONIAN_DISTANCE_EDGES_KM,
):
    """Score one seismic row, optionally using only a separate history CSV.

    A supplied history_csv is the ONLY source of parent events. Target CSV
    events are never automatically merged into that history. Thus training
    history can be frozen when evaluating another dataset.
    """
    if isinstance(row_index, bool) or not isinstance(row_index, (int, np.integer)):
        raise TypeError("row_index must be a zero-based integer.")
    if row_index < 0:
        raise IndexError("row_index must be nonnegative.")
    rows = _csv_rows(csv_filename)
    try:
        for index, target in enumerate(rows):
            if index == row_index:
                break
        else:
            raise IndexError(f"row_index {row_index} is outside the CSV data rows.")
    finally:
        rows.close()
    if not _is_seismic(target):
        raise ValueError(f"Row {row_index} is not a seismic row.")
    try:
        lat, lng = _coordinates(target)
        time = _parse_time(target["time"])
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Row {row_index} has invalid time or coordinates.") from exc
    history = load_events_from_csv(csv_filename if history_csv is None else history_csv)
    return TheCharlesonian(
        history, lat, lng, time, lookback_days=lookback_days, mag_min=mag_min,
        distance_edges_km=distance_edges_km,
    )


# CHAT: Export four historical features while preserving all original CSV rows and values.
def add_charlesonian_to_csv(
    input_csv, output_csv, *, history_csv=None, lookback_days=CHARLESONIAN_LOOKBACK_DAYS,
    mag_min=None, distance_edges_km=CHARLESONIAN_DISTANCE_EDGES_KM,
):
    """Append magnitude sums to seismic rows; other sources receive blanks.

    History defaults to input_csv, but EVERY target uses strictly earlier
    timestamps. Set history_csv to the training catalog to freeze history
    for validation/test targets. Existing matching feature columns are replaced,
    not duplicated. Invalid target coordinates/time abort the export; malformed
    historical seismic rows are skipped by the existing CSV history loader.
    Returns the number of scored seismic rows. No model fitting is performed.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    history_csv = input_csv if history_csv is None else Path(history_csv)
    if output_csv.resolve() in {input_csv.resolve(), history_csv.resolve()}:
        raise ValueError("Output must differ from both the target and history input files.")
    edges, columns = _charlesonian_columns(distance_edges_km)
    catalog = _CharlesonianCatalog(load_events_from_csv(history_csv))
    # CHAT: Validate options even when the target file has no seismic rows.
    TheCharlesonian([], 0.0, 0.0, 0.0, distance_edges_km=edges,
                    lookback_days=lookback_days, mag_min=mag_min)
    temporary_path = None
    count = 0
    try:
        with input_csv.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
            fieldnames = [name for name in reader.fieldnames if name not in columns] + columns
            # CHAT: Replace the output only after a complete successful export.
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", dir=output_csv.parent,
                suffix=".csv", delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=fieldnames)
                writer.writeheader()
                for index, row in enumerate(reader):
                    row.update(dict.fromkeys(columns, ""))
                    if _is_seismic(row):
                        try:
                            lat, lng = _coordinates(row)
                            row.update(TheCharlesonian(
                                catalog, lat, lng, _to_day(row["time"]),
                                distance_edges_km=edges, lookback_days=lookback_days, mag_min=mag_min,
                            ))
                        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                            raise ValueError(f"Seismic data row {index}: {exc}") from exc
                        count += 1
                    writer.writerow(row)
        os.replace(temporary_path, output_csv)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return count


# CHAT: Expose the new controls in the CLI; the CSV batch wrapper can continue using defaults.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_filename", help="Path to the mixed-source CSV")
    # CHAT: Row index is optional only for a whole-file Charlesonian export.
    parser.add_argument("row_index", type=int, nargs="?", help="Zero-based data-row index, excluding header")
    parser.add_argument("--dt-max", type=float, default=180.0, help="Lookback in days")
    parser.add_argument("--dist-max", type=float, default=200.0, help="Radius in km")
    parser.add_argument("--mag-min", type=float, default=1.75)
    parser.add_argument("--cell-size", type=float, default=0.5, help="Grid cell size in degrees")
    # CHAT: Expose forecast horizon separately from the historical lookback.
    parser.add_argument("--forecast-days", type=float, default=30.0)
    # CHAT: Do not implicitly change the published magnitude reference when changing --mag-min.
    parser.add_argument("--magnitude-reference", type=float, default=5.0)
    # CHAT: Make quadrature convergence tolerance and work limit configurable.
    parser.add_argument("--quadrature-rtol", type=float, default=1e-6)
    parser.add_argument("--quadrature-max-order", type=int, default=128)
    # CHAT: Add single-row and whole-file Charlesonian modes without changing the default ETAS mode.
    parser.add_argument("--charlesonian", action="store_true", help="Return magnitude sums for one row")
    parser.add_argument("--charlesonian-output", help="Write all seismic-row magnitude sums to this CSV")
    parser.add_argument("--history-csv", help="Use ONLY this CSV as Charlesonian history")
    # CHAT: Resolve an omitted history-days to 1,825 only in Charlesonian modes.
    parser.add_argument("--history-days", type=float, help="Charlesonian lookback in days; default 1825")
    parser.add_argument("--history-mag-min", type=float, help="Optional strict Charlesonian magnitude cutoff")
    args = parser.parse_args()
    # CHAT: Keep ETAS-only option validation unchanged while applying the new Charlesonian default.
    charlesonian_days = CHARLESONIAN_LOOKBACK_DAYS if args.history_days is None else args.history_days
    # CHAT: Reject ambiguous modes instead of ignoring a supplied row index or history option.
    if args.charlesonian_output and (args.row_index is not None or args.charlesonian):
        parser.error("Use --charlesonian-output without row_index or --charlesonian.")
    if not args.charlesonian_output and args.row_index is None:
        parser.error("Provide a row_index or --charlesonian-output.")
    if not (args.charlesonian or args.charlesonian_output) and any(
        option is not None for option in (args.history_csv, args.history_days, args.history_mag_min)
    ):
        parser.error("History options require a Charlesonian mode.")
    try:
        # CHAT: Export uses a prepared history once and filters earlier timestamps for every target.
        if args.charlesonian_output:
            count = add_charlesonian_to_csv(
                args.csv_filename, args.charlesonian_output, history_csv=args.history_csv,
                lookback_days=charlesonian_days, mag_min=args.history_mag_min,
            )
            print(f"Added Charlesonian features for {count:,} seismic rows: {args.charlesonian_output}")
            return
        # CHAT: A single-row Charlesonian request returns only the four magnitude sums.
        if args.charlesonian:
            result = charlesonian_from_csv_row(
                args.csv_filename, args.row_index, history_csv=args.history_csv,
                lookback_days=charlesonian_days, mag_min=args.history_mag_min,
            )
            print(json.dumps(result, indent=2))
            return
        result = etas_from_csv_row(
            args.csv_filename, args.row_index, dt_max=args.dt_max,
            dist_max=args.dist_max, mag_min=args.mag_min, cell_size=args.cell_size,
            # CHAT: Forward the new CLI settings to the selected-row calculation.
            forecast_days=args.forecast_days, magnitude_reference=args.magnitude_reference,
            quadrature_rtol=args.quadrature_rtol, quadrature_max_order=args.quadrature_max_order,
        )
    except (OSError, ValueError, IndexError, ArithmeticError) as exc:  # CHAT: Report quadrature failures clearly.
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
