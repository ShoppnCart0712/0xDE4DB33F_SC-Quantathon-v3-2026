"""Append ETAS to seismic rows using the previously updated equations file.

Requires NumPy (used by the equations file).
Place this script beside the UPDATED fuckassequation.py, then run:
    python add_etas_to_csv.py "earthquakeq_train.csv" "earthquakeq_train_with_etas.csv"

Use --equations PATH if the updated equations file is elsewhere.
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
import importlib.util
import math
import os
from pathlib import Path
import tempfile


def add_etas_to_csv(
    input_csv, output_csv, equations_file=None, *,
    dt_max=180.0, dist_max=200.0, mag_min=1.75, cell_size=0.5,
):
    """Write a CSV with ETAS as its final column; return the seismic row count.

    Original fields and row order are preserved. Non-seismic rows get a blank
    ETAS value. An existing ETAS column is recalculated, not duplicated.
    ETAS is the original probability-style score, 1 - exp(-total_lambda).

    Qualifying history has 0 < time difference < dt_max (days), distance
    < dist_max (km), and magnitude > mag_min. Same-time and future events are
    excluded even if the CSV is unsorted. Defaults match the updated equations.
    Targets must be within the original latitude 25..50, longitude -90..-65 grid.
    Malformed historical events are skipped by the equations loader; an invalid
    seismic target raises an error identifying its zero-based data-row index.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError("Choose a different output path to preserve the input CSV.")
    if not math.isfinite(dt_max) or dt_max <= 0:
        raise ValueError("dt_max must be finite and positive.")
    equations_file = Path(equations_file) if equations_file else (
        Path(__file__).with_name('fuckassequation.py')
    )
    spec = importlib.util.spec_from_file_location('etas_equations', equations_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load equations from {equations_file}")
    etas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(etas)

    # Load history once, then use a time index for each target's lookback.
    # This avoids reopening and scanning the entire CSV for every target.
    events = sorted(etas.load_events_from_csv(input_csv), key=lambda e: e.day)
    days = [event.day for event in events]
    seismic_count = 0
    temporary_path = None
    try:
        with input_csv.open('r', newline='', encoding='utf-8-sig') as source:
            reader = csv.DictReader(source)
            fields = [name for name in reader.fieldnames if name != 'ETAS'] + ['ETAS']
            # Publish only a complete output if a target or calculation fails.
            with tempfile.NamedTemporaryFile(
                mode='w', newline='', encoding='utf-8',
                dir=output_csv.parent, suffix='.csv', delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                for index, row in enumerate(reader):
                    row['ETAS'] = ''
                    if etas._is_seismic(row):
                        try:
                            lat, lng = etas._coordinates(row)
                            current_day = etas._to_day(row['time'])
                            start = bisect_right(days, current_day - dt_max)
                            stop = bisect_left(days, current_day)
                            result = etas._calculate_etas(
                                events[start:stop], lat, lng, current_day,
                                dt_max=dt_max, dist_max=dist_max,
                                mag_min=mag_min, cell_size=cell_size,
                            )
                            row['ETAS'] = result['etas_score']
                        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                            raise ValueError(f"Seismic data row {index}: {exc}") from exc
                        seismic_count += 1
                    writer.writerow(row)
        os.replace(temporary_path, output_csv)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return seismic_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    parser.add_argument('--equations', help='Path to the updated ETAS equations .py')
    parser.add_argument('--dt-max', type=float, default=180.0)
    parser.add_argument('--dist-max', type=float, default=200.0)
    parser.add_argument('--mag-min', type=float, default=1.75)
    parser.add_argument('--cell-size', type=float, default=0.5)
    args = parser.parse_args()
    count = add_etas_to_csv(
        args.input_csv, args.output_csv, args.equations,
        dt_max=args.dt_max, dist_max=args.dist_max,
        mag_min=args.mag_min, cell_size=args.cell_size,
    )
    print(f'Added ETAS for {count:,} seismic rows: {args.output_csv}')
