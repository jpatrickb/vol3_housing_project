# Results

This page summarizes key empirical findings. See the paper for full detail; highlights are included here for quick review.

Clustering and regional dynamics
- Central states tend to cluster together across a range of cluster counts; coastal states (e.g., California, Florida) often form distinct small clusters.
- As the number of clusters increases (>7), singleton clusters appear frequently, suggesting diminishing returns to finer partitioning.

<img src="./assets/3clusters.png" alt="US states clustered into 3 groups" width="420"/> <img src="./assets/7clusters.png" alt="US states clustered into 7 groups" width="420"/>

Seasonality and decomposition
- Many states exhibit a double-peaked seasonal pattern within a year.
- Residual volatility increases sharply post-2020, motivating the pre-2020 evaluation window.

<img src="./assets/utah_prices.png" alt="Utah housing prices and decomposition highlights" width="720"/>

Univariate ARIMA forecasting
- Typical best model ARIMA(4,1,0). Forecasts often extend recent linear trends; performance varies widely by state.
- Predictive intervals can grow large with horizon length; about 30/51 states had test data within 95% intervals.

<img src="./assets/utah_forecast.png" alt="ARIMA forecast for Utah" width="420"/> <img src="./assets/ohio_forecast.png" alt="ARIMA forecast for Ohio" width="420"/>

VARMAX forecasting on clusters
- Cluster-based VARMAX better captures trend fluctuations than univariate ARIMA for non-trivial clusters; not applicable to singleton clusters.

<img src="./assets/utah_varmax.png" alt="VARMAX forecasting on Utah using cluster info" width="720"/>

Bayesian hierarchical model
- State-level slopes (growth rates) and intercepts (price level) differ substantially; DC and Hawaii show fastest growth; Nevada and Michigan among slowest (2000–2016).
- Standardized global covariates show correlations: higher Native/Asian/White shares associated with lower prices; higher Black share associated with rising prices (up to 2016).
- Trace plots indicate good mixing; prior predictive checks are broadly compatible with observed scales.

<img src="./assets/forest_plot.png" alt="Posterior intercepts by state" width="420"/> <img src="./assets/slope_forest.png" alt="Posterior slopes by state" width="420"/>

<img src="./assets/prior_predictive_samples.png" alt="Prior predictive distribution" width="420"/> <img src="./assets/trace_plot.png" alt="Trace plot excerpts" width="420"/>

Ethical considerations (summary)
- Demographic variables should not guide individual decisions; risk of reinforcing bias or creating harmful feedback loops.
- Better use: policy insight and regional support targeting.

