export default function Home() {
  return (
    <section className="container">
      <div className="panel" style={{padding: 24}}>
        <h1 className="site-title" style={{marginTop: 0}}>The Cost of Living: A Zillow Housing Forecast</h1>
        <p>
          This project models US state-level housing prices (2000–2020) using clustering, ARIMA/VARMAX, and a Bayesian
          hierarchical model with demographics to study regional dynamics and forecasting.
        </p>
        <h2>Explore</h2>
        <ul>
          <li><a href="#/methods">Methods</a> — formulation and modeling pipeline</li>
          <li><a href="#/results">Results</a> — selected figures and highlights</li>
          <li><a href="#/notebooks">Notebooks</a> — open notebooks on GitHub</li>
          <li><a href="#/data">Data</a> — sources and preprocessing</li>
          <li><a href="#/contributors">Contributors</a></li>
          <li><a href="/vol3_housing_project/assets/write-up.pdf" target="_blank" rel="noreferrer">Paper (PDF) ↗</a></li>
        </ul>
      </div>
    </section>
  )
}

