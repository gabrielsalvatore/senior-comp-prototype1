"""Backward-compatible entry point. The code now lives in the pie/ package.

Usage:
    python3 pie_demo.py path/to/intrvw23            (full demo)
    python3 pie_demo.py path/to/intrvw23 --quick    (skip the timing section)
    python3 -m pie path/to/intrvw23 --cpi data/cpi_u.csv
"""

from pie.cli import main

if __name__ == "__main__":
    main()
