#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Combine sequential raw CSV chunks into one raw CSV file.

The output preserves the input CSV format exactly: one header row followed by
all data rows from each chunk in natural filename order. This is intended for
raw chunk folders such as:

    raw_chunks/chunk_000001.csv
    raw_chunks/chunk_000002.csv
    ...

The function validates that every chunk has the same header and writes a small
summary text file next to the combined CSV.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PATTERN = "chunk_*.csv"
COPY_BUFFER_SIZE = 1024 * 1024 * 8
TIME_COLUMN_CANDIDATES = ("time_s", "time", "t", "seconds")


@dataclass
class ChunkInfo:
    path: Path
    first_time: float | None
    first_row: list[str] | None
    row_count: int = 0
    last_time: float | None = None
    last_row: list[str] | None = None
    ends_with_newline: bool = True


def natural_sort_key(path: str | Path) -> list[object]:
    """Sort chunk_2 before chunk_10."""
    text = Path(path).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def discover_chunk_files(input_dir: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    files = [Path(path) for path in glob.glob(str(Path(input_dir) / pattern))]
    return sorted(files, key=natural_sort_key)


def _parse_csv_line(raw_line: bytes) -> list[str]:
    text = raw_line.decode("utf-8-sig", errors="replace").rstrip("\r\n")
    return next(csv.reader([text]))


def _time_column_index(header: list[str]) -> int | None:
    lowered = {name.strip().lower(): index for index, name in enumerate(header)}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _safe_float(row: list[str] | None, index: int | None) -> float | None:
    if row is None or index is None or index >= len(row):
        return None
    try:
        return float(row[index])
    except ValueError:
        return None


def inspect_chunk(path: Path, header: list[str], time_index: int | None) -> ChunkInfo:
    with path.open("rb") as handle:
        header_line = handle.readline()
        chunk_header = _parse_csv_line(header_line)
        if chunk_header != header:
            raise ValueError(
                f"Header mismatch in {path}.\n"
                f"Expected: {header}\n"
                f"Found:    {chunk_header}"
            )
        first_data_line = handle.readline()

    first_row = _parse_csv_line(first_data_line) if first_data_line.strip() else None
    return ChunkInfo(
        path=path,
        first_time=_safe_float(first_row, time_index),
        first_row=first_row,
    )


def inspect_chunks(paths: Iterable[str | Path]) -> tuple[list[str], list[ChunkInfo], int | None]:
    chunk_paths = [Path(path) for path in paths]
    if not chunk_paths:
        raise FileNotFoundError("No chunk CSV files were found.")
    missing = [str(path) for path in chunk_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing chunk CSV file(s): {missing}")

    with chunk_paths[0].open("rb") as handle:
        header = _parse_csv_line(handle.readline())
    time_index = _time_column_index(header)
    infos = [inspect_chunk(path, header, time_index) for path in chunk_paths]
    return header, infos, time_index


def default_output_path(input_dir: str | Path) -> Path:
    input_path = Path(input_dir)
    recording_name = input_path.parent.name if input_path.name.lower() == "raw_chunks" else input_path.name
    return input_path.parent / f"{recording_name}_combined_raw.csv"


def copy_chunk_data(source_path: Path, output, info: ChunkInfo, time_index: int | None) -> None:
    """Copy a chunk's data rows and update ChunkInfo in the same pass."""
    newline_count = 0
    saw_data = False
    pending_line = b""
    last_data_line = b""
    last_byte = b""

    with source_path.open("rb") as source:
        source.readline()  # skip header
        while True:
            block = source.read(COPY_BUFFER_SIZE)
            if not block:
                break
            output.write(block)
            saw_data = True
            last_byte = block[-1:]
            newline_count += block.count(b"\n")

            pending_line += block
            lines = pending_line.split(b"\n")
            for line in lines[:-1]:
                if line.strip():
                    last_data_line = line.rstrip(b"\r")
            pending_line = lines[-1]

    if pending_line.strip():
        last_data_line = pending_line.rstrip(b"\r")

    info.ends_with_newline = (not saw_data) or last_byte == b"\n"
    info.row_count = newline_count + (1 if saw_data and not info.ends_with_newline else 0)
    info.last_row = _parse_csv_line(last_data_line) if last_data_line.strip() else info.first_row
    info.last_time = _safe_float(info.last_row, time_index)


def combine_raw_chunks(
    input_dir: str | Path,
    out_csv: str | Path | None = None,
    pattern: str = DEFAULT_PATTERN,
    chunk_files: Iterable[str | Path] | None = None,
    strict_time: bool = False,
    write_summary: bool = True,
) -> Path:
    """Combine raw chunk CSV files into one CSV with the same header/columns."""
    paths = list(chunk_files) if chunk_files is not None else discover_chunk_files(input_dir, pattern)
    paths = sorted([Path(path) for path in paths], key=natural_sort_key)
    header, infos, time_index = inspect_chunks(paths)

    destination = Path(out_csv) if out_csv is not None else default_output_path(input_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    source_paths = {info.path.resolve() for info in infos}
    if destination_resolved in source_paths:
        raise ValueError("Output path cannot be one of the input chunk files.")

    with destination.open("wb") as output:
        with infos[0].path.open("rb") as first_handle:
            header_line = first_handle.readline()
        output.write(header_line)
        if not header_line.endswith(b"\n"):
            output.write(b"\n")

        for info in infos:
            copy_chunk_data(info.path, output, info, time_index)
            if not info.ends_with_newline:
                output.write(b"\n")

    time_warnings = _time_warnings(infos, time_index)
    if write_summary:
        write_combine_summary(destination.with_suffix(destination.suffix + ".summary.txt"), header, infos, time_warnings)
    if strict_time and time_warnings:
        raise ValueError("\n".join(time_warnings))

    return destination


def _time_warnings(infos: list[ChunkInfo], time_index: int | None) -> list[str]:
    if time_index is None or len(infos) < 2:
        return []

    warnings: list[str] = []
    for previous, current in zip(infos, infos[1:]):
        if previous.first_time is None or previous.last_time is None:
            continue
        if current.first_time is None or current.last_time is None:
            continue
        if current.first_time < previous.last_time:
            warnings.append(
                f"Time decreases between {previous.path.name} ({previous.last_time}) "
                f"and {current.path.name} ({current.first_time})."
            )
    return warnings


def write_combine_summary(path: Path, header: list[str], infos: list[ChunkInfo], time_warnings: list[str]) -> None:
    total_rows = sum(info.row_count for info in infos)
    first_time = next((info.first_time for info in infos if info.first_time is not None), None)
    last_time = next((info.last_time for info in reversed(infos) if info.last_time is not None), None)

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Raw chunk combine summary\n")
        handle.write("=========================\n\n")
        handle.write(f"Chunks combined: {len(infos)}\n")
        handle.write(f"Total data rows: {total_rows}\n")
        handle.write(f"Columns: {', '.join(header)}\n")
        if first_time is not None and last_time is not None:
            handle.write(f"Time start (s): {first_time:.6f}\n")
            handle.write(f"Time end (s): {last_time:.6f}\n")
            handle.write(f"Duration span (s): {last_time - first_time:.6f}\n")
        handle.write("\nChunks\n")
        handle.write("------\n")
        for info in infos:
            handle.write(
                f"{info.path.name}: rows={info.row_count}, "
                f"first_time={info.first_time}, last_time={info.last_time}\n"
            )
        if time_warnings:
            handle.write("\nTime warnings\n")
            handle.write("-------------\n")
            for warning in time_warnings:
                handle.write(f"{warning}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine raw CSV chunk files into one raw CSV.")
    parser.add_argument("input_dir", help="Folder containing chunk CSV files, usually raw_chunks.")
    parser.add_argument("--out-csv", help="Output combined CSV path. Defaults next to raw_chunks.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"Chunk glob pattern. Default: {DEFAULT_PATTERN}")
    parser.add_argument("--strict-time", action="store_true", help="Fail if chunk time decreases between files.")
    parser.add_argument("--no-summary", action="store_true", help="Do not write the .summary.txt file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = combine_raw_chunks(
        input_dir=args.input_dir,
        out_csv=args.out_csv,
        pattern=args.pattern,
        strict_time=args.strict_time,
        write_summary=not args.no_summary,
    )
    print(f"Combined raw CSV saved: {output}")


if __name__ == "__main__":
    main()
