"""Plain-text tables for command output."""

from __future__ import annotations


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def format_table(rows: list[dict], columns: list[str]) -> str:
    cells = [columns] + [[_cell(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    lines = ("  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells)
    return "\n".join(line.rstrip() for line in lines)
