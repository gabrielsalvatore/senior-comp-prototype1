"""Feasibility check for the senior comp: CE Interview microdata.

Usage:
    1. Download a CE Interview CSV package (e.g. 2023) from
       https://www.bls.gov/cex/pumd_data.htm and unzip it.
    2. python feasibility_check.py path/to/unzipped/folder [path/to/concordance.csv]

The optional concordance CSV needs a UCC column and an ELI column
(copy the table from https://www.bls.gov/cpi/additional-resources/ce-cpi-concordance.htm).

Requires: pandas
"""

import sys
from pathlib import Path

import pandas as pd

REQUIRED_FMLI = (
    ["NEWID", "FINLWT21", "FINCBTXM", "FAM_SIZE", "AGE_REF", "REGION", "BLS_URBN",
     "SEX_REF", "FAM_TYPE", "EDUC_REF", "REF_RACE", "HISP_REF", "CUTENURE", "NO_EARNR"]
    + [f"WTREP{i:02d}" for i in range(1, 45)]
)
REQUIRED_MTBI = ["NEWID", "UCC", "COST", "REF_MO", "REF_YR"]
HOUSING_HINTS = ("RENTEQ", "RENDWE", "RNTEQ")  # rental-equivalence candidates


def load(files):
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.upper() for c in df.columns]
        df["_FILE"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def ok(flag):
    return "OK " if flag else "MISSING"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    fmli_files = sorted(root.rglob("fmli*.csv"))
    mtbi_files = sorted(root.rglob("mtbi*.csv"))
    print(f"Found {len(fmli_files)} FMLI files, {len(mtbi_files)} MTBI files\n")
    if not fmli_files or not mtbi_files:
        sys.exit("Could not find fmli*.csv / mtbi*.csv under that folder.")

    # 1. Household file variables
    fmli = load(fmli_files)
    print("== 1. Household (FMLI) variables ==")
    missing = [c for c in REQUIRED_FMLI if c not in fmli.columns]
    for c in REQUIRED_FMLI[:14]:
        print(f"  [{ok(c in fmli.columns)}] {c}")
    reps = [c for c in REQUIRED_FMLI if c.startswith("WTREP")]
    n_reps = sum(c in fmli.columns for c in reps)
    print(f"  [{ok(n_reps == 44)}] replicate weights WTREP01-44 ({n_reps}/44 found)")
    housing = [c for c in fmli.columns if any(h in c for h in HOUSING_HINTS)]
    print(f"  [{ok(bool(housing))}] rental-equivalence candidates: {housing or 'none'}")

    # 2. Sample sizes, especially rural
    print("\n== 2. Sample sizes ==")
    print(f"  household-quarter records: {len(fmli):,}")
    print(f"  unique households (NEWID // 10): {fmli['NEWID'].floordiv(10).nunique():,}")
    if "BLS_URBN" in fmli.columns:
        counts = fmli["BLS_URBN"].value_counts(dropna=False).sort_index()
        wshare = fmli.groupby("BLS_URBN")["FINLWT21"].sum() / fmli["FINLWT21"].sum()
        print("  BLS_URBN (1 = urban, 2 = rural expected):")
        for val, n in counts.items():
            share = wshare.get(val, float("nan"))
            print(f"    {val}: {n:,} records, {share:.1%} of weighted population")
        print("  Per quarter file:")
        print(fmli.groupby(["_FILE", "BLS_URBN"]).size().unstack(fill_value=0).to_string())

    for col in ("SEX_REF", "FAM_TYPE", "CUTENURE"):
        if col in fmli.columns:
            print(f"  {col} record counts (per year of 4 quarters):")
            print(fmli[col].value_counts(dropna=False).sort_index().to_string())

    # 3. Expenditure file
    mtbi = load(mtbi_files)
    print("\n== 3. Expenditure (MTBI) variables ==")
    for c in REQUIRED_MTBI:
        print(f"  [{ok(c in mtbi.columns)}] {c}")
    mtbi["UCC"] = mtbi["UCC"].astype(str).str.zfill(6)
    spend = mtbi[mtbi["COST"] > 0]
    print(f"  expenditure rows: {len(mtbi):,}; distinct UCC codes: {mtbi['UCC'].nunique():,}")
    print("  distinct months covered per year:")
    print(spend.groupby("REF_YR")["REF_MO"].nunique().to_string())

    # 4. Concordance coverage (optional)
    if len(sys.argv) > 2:
        conc = pd.read_csv(sys.argv[2], dtype=str)
        conc.columns = [c.upper().strip() for c in conc.columns]
        conc["UCC"] = conc["UCC"].str.strip().str.zfill(6)
        mapped = spend["UCC"].isin(set(conc["UCC"]))
        cov = spend.loc[mapped, "COST"].sum() / spend["COST"].sum()
        print("\n== 4. Concordance coverage ==")
        print(f"  share of spending ($) whose UCC maps to a CPI item: {cov:.1%}")
        top = (spend.loc[~mapped].groupby("UCC")["COST"].sum()
               .sort_values(ascending=False).head(10))
        print("  largest unmapped UCCs by spending:")
        print(top.to_string())

    print("\nSummary: missing FMLI variables ->", missing or "none")


if __name__ == "__main__":
    main()
