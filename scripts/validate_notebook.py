"""Validate that a committed Jupyter notebook is readable and error-free."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise SystemExit("expected notebook format 4")

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise SystemExit(f"notebook contains {len(errors)} saved error output(s)")

    executed = sum(cell.get("execution_count") is not None for cell in code_cells)
    print(f"{args.notebook}: {len(code_cells)} code cells, {executed} executed, 0 errors")


if __name__ == "__main__":
    main()
