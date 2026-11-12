"""Category definitions and survey constants shared by every module."""

from __future__ import annotations

from pathlib import Path

# Default CPI file: CPI-U, U.S. city average, not seasonally adjusted (BLS via FRED).
DEFAULT_CPI_FILE = Path(__file__).resolve().parent.parent / "data" / "cpi_u.csv"

# The 16 CPI categories, in the column order used for the H x I weight matrix.
CATEGORIES = [
    "Food at home", "Food away from home", "Alcohol", "Shelter",
    "Utilities", "Household furnishings", "Apparel", "Vehicles",
    "Gasoline", "Transport services", "Medical care", "Recreation",
    "Education", "Personal care", "Tobacco", "Other goods/services",
]

# CE summary variables per category. Each stem is read for the current
# quarter (CQ) and the previous quarter (PQ). Shelter is built separately.
CE_VARS = {
    "Food at home": ["FDHOME"],
    "Food away from home": ["FDAWAY"],
    "Alcohol": ["ALCBEV"],
    "Utilities": ["UTIL"],
    "Household furnishings": ["HOUSOP", "HOUSEQ"],
    "Apparel": ["APPAR"],
    "Vehicles": ["CARTKN", "CARTKU", "OTHVEH"],
    "Gasoline": ["GASMO"],
    "Transport services": ["VEHFIN", "MAINRP", "VEHINS", "VRNTLO", "PUBTRA"],
    "Medical care": ["HEALTH"],
    "Recreation": ["ENTERT", "READ"],
    "Education": ["EDUCA"],
    "Personal care": ["PERSCA"],
    "Tobacco": ["TOBACC"],
    "Other goods/services": ["MISC"],
}
QUARTER_SUFFIXES = ("CQ", "PQ")

# Shelter inputs: rent for renters, rental equivalence for owners, other lodging.
SHELTER_VARS = ["RENDWECQ", "RENDWEPQ", "OTHLODCQ", "OTHLODPQ",
                "OWNDWECQ", "OWNDWEPQ", "RENTEQVX", "CUTENURE"]
OWNER_TENURE_CODES = [1, 2, 3]

# Number of BLS balanced-repeated-replication (BRR) replicate weights.
N_REP = 44
REP_COLS = [f"WTREP{i:02d}" for i in range(1, N_REP + 1)]

# Household characteristics used for grouping (taken from the last interview).
DEMOGRAPHIC_VARS = ["FAM_SIZE", "AGE_REF", "SEX_REF", "FAM_TYPE",
                    "BLS_URBN", "CUTENURE", "REGION"]
ID_AND_WEIGHT_VARS = ["NEWID", "FINLWT21", "FINCBTXM"]
