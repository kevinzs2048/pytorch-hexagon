# SPDX-License-Identifier: BSD-3-Clause
"""Validate and summarize a serial Hexagon-MLIR LWP capture.

Input uses the upstream {"entries": [{"id": ..., "cyc": ...}]} format.
Nested inclusive durations overlap; exclusive cycles subtract recorded children.
This format has no thread IDs and must not be used for concurrent writers.
"""
import argparse
import csv
import json
from pathlib import Path
import re


def summarize(entries):
    stack, summary = [], {}
    previous = -1
    for entry in entries:
        site, cycle = entry['id'], entry['cyc']
        if not isinstance(site, int) or site < 1 or not isinstance(cycle, int) or cycle < previous:
            raise ValueError('Invalid site ID or non-monotonic cycle timestamp')
        previous = cycle
        if stack and stack[-1][0] == site:
            _, start, children = stack.pop()
            elapsed = cycle - start
            if elapsed < children:
                raise ValueError('Child durations exceed parent duration')
            row = summary.setdefault(site, dict(id=site, calls=0, inclusive_cycles=0,
                                                 exclusive_cycles=0, parents=set()))
            row['calls'] += 1
            row['inclusive_cycles'] += elapsed
            row['exclusive_cycles'] += elapsed - children
            row['parents'].add(stack[-1][0] if stack else 0)
            if stack:
                stack[-1][2] += elapsed
        else:
            if any(frame[0] == site for frame in stack):
                raise ValueError(f'Interleaved/reentrant site {site}; serial capture required')
            stack.append([site, cycle, 0])
    if stack:
        raise ValueError(f'Truncated capture: unclosed sites {[f[0] for f in stack]}')
    if not summary:
        raise ValueError('Empty capture')
    return [dict(row, parents=sorted(row['parents'])) for row in summary.values()]


def read_map(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'Location (.*?) corresponds to ID (\d+) \| Collected ops: ?(.*)', line)
        if not match:
            raise ValueError(f'Invalid LWP map line: {line}')
        loc, site, ops = match.groups()
        site = int(site)
        if site in result:
            raise ValueError(f'Duplicate site ID {site}; compile with MLIR threading disabled')
        result[site] = {'ir_lines':loc, 'operations':ops}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='CSV output; refuses overwrite')
    args = parser.parse_args()
    data = json.loads(args.capture.read_text())
    rows = summarize(data['entries'])
    locations = read_map(args.map)
    for row in rows:
        if row['id'] not in locations:
            raise ValueError(f"Site {row['id']} missing from this build's map")
        row.update(locations[row['id']])
    rows.sort(key=lambda r:r['exclusive_cycles'], reverse=True)
    with args.output.open('x',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Validated {len(data["entries"])} records, {len(rows)} sites: {args.output}')


if __name__ == '__main__':
    main()
