# Data

Sources
- Zillow Home Price Index (HPI): monthly state-level median home prices from January 2000 onward.
- Current Population Survey (CPS) via IPUMS: annual demographic composition, income, and property tax.

Merging and preprocessing
- Housing data: isolated missing months interpolated linearly; early long gaps (e.g., North Dakota) filled with first observed value and flagged.
- CPS data: restricted to major race categories (White, Black, Asian, Native American) due to sparsity; linearly interpolated to monthly cadence to align with HPI.
- Merge keys: month and state. Post-2020 data excluded to avoid COVID-19 shock regime.

Modeling splits
- Train: through January 2016 (80%). Test: February 2016–January 2020 (20%). Chronological split to prevent leakage.

Files and paths
- Expected inputs:
  - cps_data.dta at the project root (or update paths in src/data_loader.py)
  - Data_Files/price_by_state_cleaned.csv (Zillow HPI)
- Generated output:
  - Data_Files/state_full.csv (created by `python -m src.data_loader`)

Reproducibility notes
- Large/raw data files are ignored by Git (.gitignore). Provide paths locally before running notebooks.
- Demographic effects reflect pre-2020 relationships and may not generalize to shock periods.

