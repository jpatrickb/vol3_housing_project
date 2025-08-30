---
title: Housing Markets Project
---

# Housing Markets Project

<img src="./assets/hero.png" alt="Hero figure" width="720"/>

This project models US state-level housing prices (2000–2020) to identify regional patterns and assess forecasting techniques. We combine time-series clustering, classical decomposition, ARIMA/VARMAX forecasting, and a Bayesian hierarchical model that incorporates demographics, income, and property tax. We analyze pre-COVID dynamics (through Jan 2020) to avoid pandemic-driven shocks.

Key takeaways
- Central US states tend to share similar growth dynamics; coastal states often behave as distinct clusters.
- Univariate ARIMA forecasts are often approximately linear and sensitive to shocks; cluster-based VARMAX better captures trend fluctuations when clusters are non-trivial.
- A Bayesian hierarchical model yields interpretable state-level slopes/intercepts and global covariate effects, suggesting demographic composition correlates with price trends up to 2016.

Explore
- Methods: see [Methods](./methods.md)
- Results: see [Results](./results.md)
- Notebooks: see [Notebooks](./notebooks.md)
- Data notes: see [Data](./data.md)
- Contributors: see [Contributors](./contributors.md)
- Paper (PDF): [Download](./paper.pdf)

