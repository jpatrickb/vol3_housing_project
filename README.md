# Housing Markets: Time Series, Clustering, and Bayesian Hierarchies

This project explores US state-level housing markets with time-series modeling (ARIMA/Kalman), clustering, and Bayesian hierarchical modeling. We study forecasting accuracy, regional co-movements, and the role of demographics.

- Live site (GitHub Pages): enable in repository Settings → Pages and set source to /docs
- Methods: ARIMA/Kalman, clustering of state-level series, Bayesian hierarchies
- Highlights: prediction quality, regional clusters, demographic sensitivity
- Notebooks: see notebooks/ or the site’s Notebooks page

## Repository Structure
- notebooks/: analysis notebooks
- src/: data utilities (e.g., data_loader.py)
- assets/figures/: curated figures used in docs
- reports/: PDFs (presentation, writeups)
- docs/: GitHub Pages content

## Data
This project uses CPS (.dta) and Zillow’s state-level price data (.csv). Place:
- cps_data.dta at project root (or adjust paths)
- Data_Files/price_by_state_cleaned.csv as source

Generate processed data (creates Data_Files/state_full.csv):

```bash
python -m src.data_loader
```

## Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License
MIT

