import { Hero, TopicCard, DownloadCard } from '../components';

export default function Home() {
  return (
    <>
      <Hero
        title="Housing Markets Analysis"
        subtitle="State-Level Forecasting with Demographics"
        author="A Time-Series Clustering and Bayesian Approach"
        buttons={[
          { label: 'View Results', href: '#/results', icon: '📊', variant: 'primary' },
          { label: 'Read Methods', href: '#/methods', icon: '📚', variant: 'secondary' },
        ]}
      />

      <section className="overview">
        <div className="container">
          <h2>Project Overview</h2>
          <p className="intro-text">
            This project models US state-level housing prices (2000–2020) using clustering, ARIMA/VARMAX, and a Bayesian hierarchical model with demographics to study regional dynamics and forecasting.
          </p>
          <div className="grid-2">
            <div className="card">
              <h3>Methodology</h3>
              <p>
                We combine classical time-series decomposition, K-means clustering, ARIMA/VARMAX forecasting, and Bayesian hierarchical modeling to understand housing market behavior across US states.
              </p>
              <p style={{ marginTop: '1rem' }}>
                <a href="#/methods" style={{ fontWeight: 600 }}>Explore Methods →</a>
              </p>
            </div>
            <div className="card">
              <h3>Key Findings</h3>
              <p>
                Central states cluster together while coastal states form distinct groups. VARMAX on clusters improves forecasting, and hierarchical models reveal demographic effects on housing growth rates.
              </p>
              <p style={{ marginTop: '1rem' }}>
                <a href="#/results" style={{ fontWeight: 600 }}>View Results →</a>
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="topics-section">
        <div className="container">
          <h2>Analysis Components</h2>
          <div className="topics-grid">
            <TopicCard
              icon="🗂️"
              title="Data Integration"
              description="Merged Zillow home prices with CPS demographics"
              items={[
                'Monthly state-level housing prices (2000–2020)',
                'Annual demographic composition and income',
                '51 regions (50 states + DC)',
                'Linear interpolation for alignment',
              ]}
            />
            <TopicCard
              icon="📈"
              title="Clustering & Forecasting"
              description="Regional patterns and time-series modeling"
              items={[
                'K-means clustering (2–20 clusters)',
                'ARIMA univariate forecasting',
                'VARMAX multivariate models on clusters',
                'Out-of-sample forecasting (2016–2020)',
              ]}
            />
            <TopicCard
              icon="🔬"
              title="Bayesian Hierarchical Model"
              description="State-level effects with demographic covariates"
              items={[
                'Linear regression with hierarchical priors',
                'State-specific intercepts and slopes',
                'Global effects: tax, population, race',
                'PyMC3 NUTS sampling (8k draws)',
              ]}
            />
          </div>
        </div>
      </section>

      <section className="download">
        <div className="container">
          <h2>Resources</h2>
          <p style={{ textAlign: 'center', maxWidth: '900px', margin: '0 auto 2rem', color: '#404040' }}>
            Access the complete project including the full write-up, all Jupyter notebooks, and source code on GitHub.
          </p>
          <div className="download-grid">
            <DownloadCard
              icon="📑"
              title="Full Write-Up"
              description="Comprehensive paper with methodology, results, and detailed analysis of housing market forecasting."
              href="/vol3_housing_project/assets/write-up.pdf"
              ctaText="Read PDF →"
            />
            <DownloadCard
              icon="💻"
              title="Jupyter Notebooks"
              description="Interactive notebooks for data processing, analysis, and model development. Fully reproducible and well-commented."
              href="#/notebooks"
              ctaText="View Notebooks →"
            />
            <DownloadCard
              icon="🔗"
              title="GitHub Repository"
              description="Complete source code, data processing scripts, and development history for the entire project."
              href="https://github.com/jpatrickb/vol3_housing_project"
              ctaText="Visit Repository →"
            />
          </div>
        </div>
      </section>
    </>
  );
}

