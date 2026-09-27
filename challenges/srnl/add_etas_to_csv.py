# CHAT: Restore rolling five-year Charlesonian history, retaining both ETAS parameter modes.
"""Append ETAS and four Charlesonian magnitude sums to seismic CSV rows.

Requires NumPy and fuckassequation.py beside this file.
Example (from challenges/srnl):
    python add_etas_to_csv.py Data/earthquakeq_test.csv Data/earthquakeq_test_features.csv --history-csv Data/earthquakeq_all_history.csv

# CHAT: Select paper (0) or fitted (1) amplitudes and name outputs automatically.
    python add_etas_to_csv.py Data/earthquakeq_test.csv --fitted 0 --history-csv Data/earthquakeq_all_history.csv
    python add_etas_to_csv.py Data/earthquakeq_test.csv --fitted 1 --history-csv Data/earthquakeq_all_history.csv
With no explicit output path, 0 writes earthquakeq_test_featured.csv and 1
writes earthquakeq_test_feature_fitted.csv in the input file's directory.
# CHAT: Fitted mode uses hard-coded K=0.846192771684 and mu=2.12608430449e-08.
Only mu and K change; no JSON file is required. Charlesonian history and
startup behavior are identical in both modes.

Each row uses history strictly before its own timestamp. A merged history
catalog is suitable for rolling forecasts only when those earlier events
would already have been observed. Use --history-before TIME for forecasts
whose information is frozen at a common issuance time. No model is fitted.

Charlesonian uses only the preceding 1,825 days, all valid magnitudes, and four
disjoint bands. Early rows use the available partial history and have
Charlesonian_full_window=0; after a full window is available the flag becomes 1.
ETAS keeps its own defaults: 180 days, 200 km, magnitude >1.75.
--charlesonian-days changes the rolling lookback (default 1825 days).
--charlesonian-warmup-days optionally withholds early values (default 0).
"""

import argparse
from bisect import bisect_left, bisect_right
import csv
import importlib.util
import math
import os
from pathlib import Path
import tempfile


# CHAT: Give paper and fitted feature datasets distinct automatic names, retaining explicit output paths.
def feature_output_path(input_csv, fitted=0):
    if isinstance(fitted, bool) or not isinstance(fitted, int) or fitted not in (0, 1):
        raise ValueError('fitted must be integer 0 (paper) or 1 (fitted mu/K).')
    path = Path(input_csv)
    suffix = '_feature_fitted' if fitted else '_featured'
    return path.with_name(path.stem + suffix + '.csv')


# CHAT: Preserve the original callable and return value, adding shared-history and Charlesonian options.
def add_etas_to_csv(
    input_csv, output_csv=None, equations_file=None, *,
    dt_max=180.0, dist_max=200.0, mag_min=1.75, cell_size=0.5,
    history_csv=None, charlesonian_days=1825.0, charlesonian_mag_min=None,  # CHAT: Restore rolling five-year default.
    history_start=None, history_before=None,
    charlesonian_warmup_days=0.0,  # CHAT: Restore partial startup values; retain an optional explicit delay.
    fitted=0, fit_file=None, magnitude_reference=5.0,  # CHAT: 0=paper, 1=hard-coded fitted mu/K; fit_file is a compatibility argument.
):
    """Write a feature CSV; return the number of seismic rows scored.

    # CHAT: The equations selector receives fitted once before processing rows.
    fitted=0 uses the paper; fitted=1 uses the fixed fitted mu/K constants.
    # CHAT: fit_file is retained but ignored; no JSON is read.
    Omitted output_csv generates
    an input-stem_featured.csv or input-stem_feature_fitted.csv filename.

    history_csv is the sole history source for BOTH ETAS and Charlesonian;
    targets are not automatically added to a separately supplied history.
    Defaults to input_csv when omitted. Both methods exclude target-time and
    future earthquakes even if the history file is unsorted.

    history_start optionally specifies the known start of usable coverage
    (ISO timestamp). Events before that start are excluded. If omitted, the
    earliest valid seismic timestamp is used as a conservative start proxy.
    Charlesonian_full_window checks whether the supplied coverage starts at or
    before the rolling window's lower boundary. It does not check network
    completeness, missing records, or magnitude detectability. Early rows have
    ETAS, partial-history Charlesonian sums, and flag 0 by default.

    history_before optionally restricts the available history to times strictly
    before a common forecast issuance time. Rows earlier than that time remain
    protected by the additional per-target timestamp filter.

    Original columns/row order are preserved. Existing feature columns are
    replaced, not duplicated. Non-seismic feature fields are blank. Invalid
    historical seismic records are skipped by the equations loader; invalid
    target coordinates/time abort the export without publishing a partial file.
    """
    # CHAT: Validate the mode even when an explicit output path is supplied.
    automatic_output = feature_output_path(input_csv, fitted)
    input_csv, output_csv = Path(input_csv), Path(output_csv) if output_csv is not None else automatic_output
    history_csv = input_csv if history_csv is None else Path(history_csv)
    if output_csv.resolve() in {input_csv.resolve(), history_csv.resolve()}:
        raise ValueError("Output must differ from the input and history CSV paths.")
    if not math.isfinite(dt_max) or dt_max <= 0:
        raise ValueError("dt_max must be finite and positive.")
    # CHAT: Allow unlimited history and validate the independent startup delay.
    if charlesonian_days is not None and (not math.isfinite(charlesonian_days) or charlesonian_days <= 0):
        raise ValueError("charlesonian_days must be finite and positive, or None for all history.")
    if not math.isfinite(charlesonian_warmup_days) or charlesonian_warmup_days < 0:
        raise ValueError("charlesonian_warmup_days must be finite and nonnegative.")
    # CHAT: Use one equations file in both modes; its selector changes only mu/K.
    equations_file = Path(equations_file) if equations_file else Path(__file__).with_name('fuckassequation.py')
    spec = importlib.util.spec_from_file_location('etas_equations', equations_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load equations from {equations_file}")
    etas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(etas)
    if not hasattr(etas, 'CHARLESONIAN_WARMUP_DAYS'):
        raise ValueError("Replace fuckassequation.py with the accompanying updated equations file.")
    # CHAT: Select fixed amplitudes once per export, with no fitted JSON lookup.
    if output_csv.resolve() in {equations_file.resolve(), Path(__file__).resolve()}:
        raise ValueError('Output cannot overwrite an equations script or this exporter.')
    if hasattr(etas, 'select_etas_parameters'):
        parameters = etas.select_etas_parameters(
            fitted, fit_file, dt_max=dt_max, dist_max=dist_max,
            mag_min=mag_min, magnitude_reference=magnitude_reference,
        )
    elif fitted == 0:
        parameters = etas.PAPER_ZONE0_PARAMETERS.copy()  # CHAT: Allow an explicit original equations file in paper mode.
    else:
        raise ValueError('Fitted mode requires fuckassequation.py containing select_etas_parameters.')

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
                         lookback_days=charlesonian_days, mag_min=charlesonian_mag_min,
                         warmup_days=charlesonian_warmup_days, history_start=coverage_start)  # CHAT: Validate startup controls.
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
                                parameters=parameters, magnitude_reference=magnitude_reference,  # CHAT: Apply selected mu/K.
                            )
                            row['ETAS'] = result['etas_score']
                            # CHAT: Only an explicit nonzero warmup suppresses early values.
                            available_until = current_day if freeze_day is None else min(current_day, freeze_day)
                            ready = (
                                charlesonian_warmup_days == 0
                                or (coverage_start is not None
                                    and available_until >= coverage_start + charlesonian_warmup_days)
                            )
                            # CHAT: Restore the full-window flag independently of whether partial sums are output.
                            required_history_days = charlesonian_days if charlesonian_days is not None else charlesonian_warmup_days
                            row['Charlesonian_full_window'] = int(
                                coverage_start is not None
                                and current_day - required_history_days >= coverage_start
                            )
                            # CHAT: Calculate partial startup histories and discard events older than the rolling window.
                            if ready:
                                row.update(etas.TheCharlesonian(
                                    charlesonian_catalog, lat, lng, current_day,
                                    lookback_days=charlesonian_days, mag_min=charlesonian_mag_min,
                                    warmup_days=charlesonian_warmup_days, history_start=coverage_start,
                                ))
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


# CHAT: Restore rolling defaults while preserving fitted/paper and optional startup controls.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv')
    parser.add_argument('output_csv', nargs='?', help='Optional explicit output; default uses _featured or _feature_fitted')
    parser.add_argument('--equations', help='Default: fuckassequation.py beside this script')
    # CHAT: Keep the binary selector; accept legacy fit-file commands without reading the path.
    parser.add_argument('--fitted', type=int, choices=(0, 1), default=0, help='0=paper mu/K, 1=fitted mu/K')
    parser.add_argument('--fit-file', help='Legacy option, ignored; fitted mu/K are hard-coded')
    parser.add_argument('--magnitude-reference', type=float, default=5.0)
    parser.add_argument('--history-csv', help='Shared seismic history for BOTH features; default input CSV')
    parser.add_argument('--charlesonian-days', type=float, default=1825.0, help='Rolling history in days; default 1825')
    parser.add_argument('--charlesonian-warmup-days', type=float, default=0.0,
                        help='Optional startup delay; default 0 (calculate early rows)')
    parser.add_argument('--charlesonian-mag-min', type=float, help='Strict magnitude cutoff; default no cutoff')
    parser.add_argument('--history-start', help='Known usable coverage start, ISO timestamp; default first seismic time')
    parser.add_argument('--history-before', help='Freeze available history strictly before this ISO timestamp')
    parser.add_argument('--dt-max', type=float, default=180.0)
    parser.add_argument('--dist-max', type=float, default=200.0)
    parser.add_argument('--mag-min', type=float, default=1.75)
    parser.add_argument('--cell-size', type=float, default=0.5)
    args = parser.parse_args()
    try:
        # CHAT: Resolve the automatic name once so the completion message reports the actual output.
        output_csv = Path(args.output_csv) if args.output_csv else feature_output_path(args.input_csv, args.fitted)
        count = add_etas_to_csv(
            args.input_csv, output_csv, args.equations,
            dt_max=args.dt_max, dist_max=args.dist_max,
            mag_min=args.mag_min, cell_size=args.cell_size,
            history_csv=args.history_csv, charlesonian_days=args.charlesonian_days,
            charlesonian_mag_min=args.charlesonian_mag_min,
            history_start=args.history_start, history_before=args.history_before,
            charlesonian_warmup_days=args.charlesonian_warmup_days,  # CHAT: Forward startup independently.
            fitted=args.fitted, fit_file=args.fit_file, magnitude_reference=args.magnitude_reference,  # CHAT: Forward mode.
        )
    except (OSError, TypeError, ValueError, IndexError, ArithmeticError) as exc:
        parser.error(str(exc))
    mode = 'fitted mu/K' if args.fitted else 'paper mu/K'  # CHAT: Report which parameters produced the dataset.
    print(f'Added ETAS ({mode}) and Charlesonian features for {count:,} seismic rows: {output_csv}')
