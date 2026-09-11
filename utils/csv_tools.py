import pathlib
import csv
from typing import Any, Optional


def read_csv_file(
    path: pathlib.Path,
    delimiter: str = ","
) -> list[dict[str, Any]]:
    with path.open("r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        contents: list[dict[str, Any]] = [row for row in reader]
    return contents


def write_csv_file(
    data: list[dict[str, Any]],
    output_file: pathlib.Path,
    fieldnames: Optional[list[str]] = None,
    delimiter: str = ","
) -> None:
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with output_file.open("w", newline="") as f:
        writer: csv.DictWriter = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for r in data:
            writer.writerow(r)