# CHAT: Export ETAS plus five-year Charlesonian features from an optional shared historical catalog.
"""Append ETAS and four Charlesonian magnitude sums to seismic CSV rows.

Requires NumPy and the updated fuckassequation.py beside this file.
Example (from challenges/srnl):
    python add_etas_to_csv.py Data/earthquakeq_test.csv Data/earthquakeq_test_features.csv --history-csv Data/earthquakeq_all_history.csv

Each row uses history strictly before its own timestamp. A merged history
catalog is suitable for rolling forecasts only when those earlier events
would already have been observed. Use --history-before TIME for forecasts
whose information is frozen at a common issuance time. No model is fitted.

Charlesonian defaults to 1,825 days, all valid magnitudes, and four disjoint
bands. ETAS keeps its own defaults: 180 days, 200 km, magnitude >1.75.
Early partial Charlesonian windows are calculated and flagged with
Charlesonian_full_window=0. Preserve their earthquakes as history, but exclude
these rows when fitting weights that require a full five-year window.
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
import importlib.util
import math
import os
from pathlib import Path
import tempfile


# CHAT: Preserve the original callable and return value, adding shared-history and Charlesonian options.
def add_etas_to_csv(
    input_csv, output_csv, equations_file=None, *,
    dt_max=180.0, dist_max=200.0, mag_min=1.75, cell_size=0.5,
    history_csv=None, charlesonian_days=1825.0, charlesonian_mag_min=None,
    history_start=None, history_before=None,
):
    """Write a feature CSV; return the number of seismic rows scored.

    history_csv is the sole history source for BOTH ETAS and Charlesonian;
    targets are not automatically added to a separately supplied history.
    Defaults to input_csv when omitted. Both methods exclude target-time and
    future earthquakes even if the history file is unsorted.

    history_start optionally specifies the known start of usable coverage
    (ISO timestamp). Events before that start are excluded. If omitted, the
    earliest valid seismic timestamp is used as a conservative start proxy.
    Charlesonian_full_window checks only elapsed coverage time, not network
    completeness, missing records, or magnitude detectability.

    history_before optionally restricts the available history to times strictly
    before a common forecast issuance time. Rows earlier than that time remain
    protected by the additional per-target timestamp filter.

    Original columns/row order are preserved. Existing feature columns are
    replaced, not duplicated. Non-seismic feature fields are blank. Invalid
    historical seismic records are skipped by the equations loader; invalid
    target coordinates/time abort the export without publishing a partial file.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    history_csv = input_csv if history_csv is None else Path(history_csv)
    if output_csv.resolve() in {input_csv.resolve(), history_csv.resolve()}:
        raise ValueError("Output must differ from the input and history CSV paths.")
    if not math.isfinite(dt_max) or dt_max <= 0:
        raise ValueError("dt_max must be finite and positive.")
    if not math.isfinite(charlesonian_days) or charlesonian_days <= 0:
        raise ValueError("charlesonian_days must be finite and positive.")
    # CHAT: Match the user's clean filename; retain --equations for any other location.
    equations_file = Path(equations_file) if equations_file else Path(__file__).with_name('fuckassequation.py')
    spec = importlib.util.spec_from_file_location('etas_equations', equations_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load equations from {equations_file}")
    etas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(etas)
    if not hasattr(etas, 'TheCharlesonian'):
        raise ValueError("Use the updated fuckassequation.py containing TheCharlesonian.")

    # CHAT: Load shared history once and apply any explicit coverage/forecast boundaries.
    events = sorted(etas.load_events_from_csv(history_csv), key=lambda e: e.day)
    coverage_start = etas._to_day(history_start) if history_start is not None else (
        events[0].day if events else None
    )
    freeze_day = etas._to_day(history_before) if history_before is not None else None
    if coverage_start is not None and freeze_day is not None and freeze_day <= coverage_start:
        raise ValueError("history_before must be after history_start.")
    events = [event for event in events
              if (coverage_start is None or event.day >= coverage_start)
              and (freeze_day is None or event.day < freeze_day)]
    days = [event.day for event in events]
    charlesonian_catalog = etas._CharlesonianCatalog(events)
    _, band_columns = etas._charlesonian_columns(etas.CHARLESONIAN_DISTANCE_EDGES_KM)
    # CHAT: Four separate band columns retain the information needed to learn distinct weights.
    feature_columns = ['ETAS', *band_columns, 'Charlesonian_full_window']
    etas.TheCharlesonian([], 0.0, 0.0, 0.0,
                         lookback_days=charlesonian_days, mag_min=charlesonian_mag_min)
    seismic_count = 0
    temporary_path = None
    try:
        with input_csv.open('r', newline='', encoding='utf-8-sig') as source:
            reader = csv.DictReader(source)
            missing = etas.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
            fields = [name for name in reader.fieldnames if name not in feature_columns] + feature_columns
            with tempfile.NamedTemporaryFile(
                mode='w', newline='', encoding='utf-8',
                dir=output_csv.parent, suffix='.csv', delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=fields)
                writer.writeheader()
                for index, row in enumerate(reader):
                    row.update(dict.fromkeys(feature_columns, ''))
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
                            # CHAT: Use the five-year history independently of ETAS's shorter cutoff.
                            row.update(etas.TheCharlesonian(
                                charlesonian_catalog, lat, lng, current_day,
                                lookback_days=charlesonian_days, mag_min=charlesonian_mag_min,
                            ))
                            # CHAT: Flag partial startup windows without deleting their records.
                            row['Charlesonian_full_window'] = int(
                                coverage_start is not None
                                and current_day - charlesonian_days >= coverage_start
                            )
                        except (TypeError, ValueError, AttributeError, ArithmeticError) as exc:
                            raise ValueError(f"Seismic data row {index}: {exc}") from exc
                        seismic_count += 1
                    writer.writerow(row)
        os.replace(temporary_path, output_csv)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return seismic_count


# CHAT: Add CLI controls for shared history, the five-year window, coverage, and forecast issuance time.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    parser.add_argument('--equations', help='Path to fuckassequation.py; default is beside this script')
    parser.add_argument('--history-csv', help='Shared seismic history for BOTH features; default input CSV')
    parser.add_argument('--charlesonian-days', type=float, default=1825.0)
    parser.add_argument('--charlesonian-mag-min', type=float, help='Strict magnitude cutoff; default no cutoff')
    parser.add_argument('--history-start', help='Known usable coverage start, ISO timestamp; default first seismic time')
    parser.add_argument('--history-before', help='Freeze available history strictly before this ISO timestamp')
    parser.add_argument('--dt-max', type=float, default=180.0)
    parser.add_argument('--dist-max', type=float, default=200.0)
    parser.add_argument('--mag-min', type=float, default=1.75)
    parser.add_argument('--cell-size', type=float, default=0.5)
    args = parser.parse_args()
    try:
        count = add_etas_to_csv(
            args.input_csv, args.output_csv, args.equations,
            dt_max=args.dt_max, dist_max=args.dist_max,
            mag_min=args.mag_min, cell_size=args.cell_size,
            history_csv=args.history_csv, charlesonian_days=args.charlesonian_days,
            charlesonian_mag_min=args.charlesonian_mag_min,
            history_start=args.history_start, history_before=args.history_before,
        )
    except (OSError, TypeError, ValueError, IndexError, ArithmeticError) as exc:
        parser.error(str(exc))
    print(f'Added ETAS and Charlesonian features for {count:,} seismic rows: {args.output_csv}')
