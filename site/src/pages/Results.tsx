import { Hero, FindingCard, ImageGallery } from '../components';

export default function Results() {
  return (
    <>
      <Hero
        title="Results & Findings"
        subtitle="Empirical Evidence from State-Level Analysis"
      />

      <section className="findings">
        <div className="container">
          <h2>Key Findings</h2>
          <div className="findings-grid">
            <FindingCard
              icon="🗺️"
              title="Regional Clustering"
              findings={[
                { label: 'Pattern', text: 'Central states group together consistently; coastal states form distinct clusters' },
                { label: 'Insight', text: 'Geographic proximity influences market dynamics, but coastal premium creates separate market structures' },
              ]}
            />
            <FindingCard
              icon="📊"
              title="Seasonal Patterns"
              findings={[
                { label: 'Observation', text: 'Double-peaked seasonality within a year across multiple states' },
                { label: 'Implication', text: 'Post-2020 residuals became erratic, validating pre-2020 forecasting window' },
              ]}
            />
            <FindingCard
              icon="📈"
              title="Forecasting Performance"
              findings={[
                { label: 'VARMAX', text: 'Multivariate models improved trend tracking versus univariate ARIMA on clusters' },
                { label: 'Uncertainty', text: 'Wide prediction intervals reflect genuine uncertainty in housing market dynamics' },
              ]}
            />
            <FindingCard
              icon="🔬"
              title="Hierarchical Effects"
              findings={[
                { label: 'State Effects', text: 'Significant heterogeneity in intercepts and slopes across states' },
                { label: 'Demographics', text: 'Tax, population, and race composition show global effects on housing growth' },
              ]}
            />
          </div>
        </div>
      </section>

      <section className="overview">
        <div className="container">
          <h2>Clustering and Regional Dynamics</h2>
          <p className="intro-text">
            Time-series K-means clustering reveals strong regional market structures. Central states exhibit similar price dynamics while coastal markets form distinct clusters.
          </p>
          <ImageGallery
            images={[
              {
                src: '/vol3_housing_project/assets/3clusters.png',
                alt: 'US states clustered into 3 groups',
                caption: 'Three-cluster solution showing major regional divisions',
              },
              {
                src: '/vol3_housing_project/assets/7clusters.png',
                alt: 'US states clustered into 7 groups',
                caption: 'Seven-cluster solution balancing granularity and interpretability',
              },
            ]}
            columns={2}
          />
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f8f9fa', borderLeft: '4px solid #0052cc', borderRadius: '8px' }}>
            <p style={{ margin: 0, fontSize: '0.95rem', color: '#333' }}>
              <strong>Practical Insight:</strong> Cluster convergence occurs around 7 groups; beyond this, singleton clusters proliferate with diminishing interpretability. Geographic diversification can be approximated by selecting states from distinct clusters rather than neighboring regions.
            </p>
          </div>
        </div>
      </section>

      <section className="overview">
        <div className="container">
          <h2>Seasonality and Decomposition</h2>
          <p className="intro-text">
            Classical time-series decomposition reveals strong seasonal patterns in housing prices, with states showing double-peaked seasonality within each year.
          </p>
          <ImageGallery
            images={[
              {
                src: '/vol3_housing_project/assets/utah_prices.png',
                alt: 'Utah prices and decomposition showing trend, seasonal, and residual components',
                caption: 'Utah housing price decomposition illustrating trend, seasonal, and residual patterns',
              },
            ]}
            columns={1}
          />
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f8f9fa', borderLeft: '4px solid #0052cc', borderRadius: '8px' }}>
            <p style={{ margin: 0, fontSize: '0.95rem', color: '#333' }}>
              <strong>Observation:</strong> Linear differencing proved most effective for achieving stationarity compared to moving averages or OLS detrending. The erratic post-2020 residuals validate our decision to exclude COVID-era data from the forecasting window.
            </p>
          </div>
        </div>
      </section>

      <section className="overview">
        <div className="container">
          <h2>ARIMA Forecasting</h2>
          <p className="intro-text">
            Univariate ARIMA models (typically ARIMA(4,1,0)) extend recent linear trends with variable accuracy across states. Wide prediction intervals reflect genuine uncertainty in forecasting.
          </p>
          <ImageGallery
            images={[
              {
                src: '/vol3_housing_project/assets/utah_forecast.png',
                alt: 'ARIMA forecast for Utah showing actual vs. predicted prices',
                caption: 'ARIMA(4,1,0) forecast for Utah with confidence intervals',
              },
              {
                src: '/vol3_housing_project/assets/ohio_forecast.png',
                alt: 'ARIMA forecast for Ohio showing actual vs. predicted prices',
                caption: 'ARIMA(4,1,0) forecast for Ohio illustrating variable performance',
              },
            ]}
            columns={2}
          />
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f8f9fa', borderLeft: '4px solid #0052cc', borderRadius: '8px' }}>
            <p style={{ margin: 0, fontSize: '0.95rem', color: '#333' }}>
              <strong>Result:</strong> Auto_arima search (p,q up to 12) converges to similar AR/MA orders across states. State-specific differences in forecast performance suggest that incorporating regional clustering improves overall predictions.
            </p>
          </div>
        </div>
      </section>

      <section className="overview">
        <div className="container">
          <h2>VARMAX Forecasting on Clusters</h2>
          <p className="intro-text">
            Multivariate VARMAX models fit on clustered states improve trend tracking and leverage market interdependencies within regional groups.
          </p>
          <ImageGallery
            images={[
              {
                src: '/vol3_housing_project/assets/utah_varmax.png',
                alt: 'VARMAX cluster-based forecasting for Utah showing improved predictions',
                caption: 'VARMAX forecasting with cluster-based interdependencies',
              },
            ]}
            columns={1}
          />
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f8f9fa', borderLeft: '4px solid #0052cc', borderRadius: '8px' }}>
            <p style={{ margin: 0, fontSize: '0.95rem', color: '#333' }}>
              <strong>Finding:</strong> VARMAX on clusters improves tracking of trends versus univariate ARIMA, particularly when cluster structure is non-trivial. VARMAX with demographic covariates often failed to converge but indicated potential for hybrid approaches.
            </p>
          </div>
        </div>
      </section>

      <section className="overview">
        <div className="container">
          <h2>Bayesian Hierarchical Model</h2>
          <p className="intro-text">
            Hierarchical Bayesian regression with state-specific effects and global demographic covariates reveals heterogeneous growth patterns while sharing information across states.
          </p>
          <h3 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Posterior Estimates by State</h3>
          <ImageGallery
            images={[
              {
                src: '/vol3_housing_project/assets/forest_plot.png',
                alt: 'Forest plot of posterior intercepts (baseline levels) for all states',
                caption: 'Posterior intercepts: baseline housing price levels by state',
              },
              {
                src: '/vol3_housing_project/assets/slope_forest.png',
                alt: 'Forest plot of posterior slopes (growth rates) for all states',
                caption: 'Posterior slopes: annual growth rates by state',
              },
            ]}
            columns={2}
          />
          <h3 style={{ marginTop: '2rem', marginBottom: '1rem' }}>Model Diagnostics</h3>
          <ImageGallery
            images={[
              {
                src: '/vol3_housing_project/assets/prior_predictive_samples.png',
                alt: 'Prior predictive distribution showing model assumptions',
                caption: 'Prior predictive checks validating weakly informative priors',
              },
              {
                src: '/vol3_housing_project/assets/trace_plot.png',
                alt: 'Trace plots for MCMC chains showing convergence',
                caption: 'NUTS trace plots indicating good mixing and convergence',
              },
            ]}
            columns={2}
          />
          <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f8f9fa', borderLeft: '4px solid #0052cc', borderRadius: '8px' }}>
            <p style={{ margin: 0, fontSize: '0.95rem', color: '#333' }}>
              <strong>Model Details:</strong> PyMC3 NUTS sampling with 4 chains, 2k tuning + 2k draws per chain (8k total samples). Trace plots confirm good mixing and convergence. State-specific intercepts and slopes show substantial heterogeneity, while hierarchical priors enable information sharing across states—critical for small-sample state-level inference.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}

