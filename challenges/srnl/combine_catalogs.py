# CHAT: New utility to merge raw train/test/judge catalogs into chronological historical input.
"""Combine CSV catalogs with the same columns, sort by UTC time, and deduplicate.

Example (from challenges/srnl):
    python combine_catalogs.py Data/earthquakeq_train.csv Data/earthquakeq_test.csv Data/earthquakeq_judge.csv --output Data/earthquakeq_all_history.csv

All sources and original field strings are retained. Input headers may be in
different orders, but must contain the same unique column names. Use the RAW
catalogs, not mixtures of raw and feature-augmented files. Exact duplicate
records are removed by default; --keep-duplicates disables this. Deduplication
compares all fields, so it does not claim to reconcile different records of
the same physical earthquake. The first input's column order is preserved.

Sorting uses a temporary SQLite database to avoid holding the entire merged
dataset in memory. Inputs are never modified. Invalid timestamps, malformed
rows, and incompatible schemas stop the merge without publishing partial output.
The combined file is history only: keep the original train/test/judge splits
for target rows and model evaluation. Feature generation must filter history
by the information available at each forecast issuance time.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile


# CHAT: Normalize timestamps for sorting without changing the original CSV timestamp strings.
def _sort_timestamp(value):
    parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec='microseconds')


# CHAT: Use disk-backed sorting, stable ordering for ties, and exact-record duplicate removal.
def combine_catalogs(input_csvs, output_csv, *, keep_duplicates=False):
    inputs = [Path(path) for path in input_csvs]
    output = Path(output_csv)
    if not inputs:
        raise ValueError('Provide at least one input CSV.')
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError('Output must be different from every input CSV.')
    total = duplicates = 0
    columns = None
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='catalog_merge_') as temporary:
        database = Path(temporary) / 'records.sqlite'
        connection = sqlite3.connect(database)
        try:
            unique = '' if keep_duplicates else ' UNIQUE'
            connection.execute(
                f'CREATE TABLE records (seq INTEGER PRIMARY KEY, sort_time TEXT NOT NULL, payload TEXT NOT NULL{unique})'
            )
            for path in inputs:
                with path.open('r', newline='', encoding='utf-8-sig') as source:
                    reader = csv.DictReader(source)
                    header = reader.fieldnames
                    if not header or len(header) != len(set(header)) or any(not name for name in header):
                        raise ValueError(f'{path}: missing or duplicate column names.')
                    if 'source' not in header or 'time' not in header:
                        raise ValueError(f'{path}: source and time columns are required.')
                    if columns is None:
                        columns = header
                    elif set(header) != set(columns):
                        raise ValueError(f'{path}: columns differ from the first CSV; combine matching raw files.')
                    for index, row in enumerate(reader):
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(f'{path}, data row {index}: wrong number of fields.')
                        try:
                            stamp = _sort_timestamp(row['time'])
                        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                            raise ValueError(f'{path}, data row {index}: invalid time {row["time"]!r}.') from exc
                        payload = json.dumps([row[name] for name in columns], ensure_ascii=False, separators=(',', ':'))
                        cursor = connection.execute(
                            'INSERT OR IGNORE INTO records(sort_time,payload) VALUES (?,?)', (stamp, payload)
                        )
                        total += 1
                        duplicates += int(cursor.rowcount == 0)
                connection.commit()
            connection.execute('CREATE INDEX time_order ON records(sort_time,seq)')
            staged = Path(temporary) / 'combined.csv'
            with staged.open('w', newline='', encoding='utf-8') as destination:
                writer = csv.writer(destination)
                writer.writerow(columns)
                for (payload,) in connection.execute('SELECT payload FROM records ORDER BY sort_time,seq'):
                    writer.writerow(json.loads(payload))
            # CHAT: Publish atomically only after all inputs have passed validation and export is complete.
            os.replace(staged, output)
        finally:
            connection.close()
    return {'input_rows': total, 'output_rows': total - duplicates, 'duplicates_removed': duplicates}


# CHAT: Accept any number of named catalogs, including train, test, and judge.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csvs', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--keep-duplicates', action='store_true')
    args = parser.parse_args()
    try:
        stats = combine_catalogs(args.input_csvs, args.output, keep_duplicates=args.keep_duplicates)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(f"Wrote {stats['output_rows']:,} rows to {args.output}; removed {stats['duplicates_removed']:,} exact duplicates.")
