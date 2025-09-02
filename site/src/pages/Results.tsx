export default function Results() {
  return (
    <section className="container">
      <div className="panel" style={{padding:24}}>
        <h1 style={{marginTop:0}}>Results</h1>
        <p>This page summarizes key empirical findings with selected figures.</p>

        <h2>Clustering and regional dynamics</h2>
        <img src="/vol3_housing_project/assets/3clusters.png" alt="US states clustered into 3 groups" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)', marginRight:'2%'}} />
        <img src="/vol3_housing_project/assets/7clusters.png" alt="US states clustered into 7 groups" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)'}} />

        <h2>Seasonality and decomposition</h2>
        <img src="/vol3_housing_project/assets/utah_prices.png" alt="Utah prices and decomposition" style={{width:'100%', borderRadius:12, border:'1px solid var(--border)'}} />

        <h2>ARIMA forecasting</h2>
        <img src="/vol3_housing_project/assets/utah_forecast.png" alt="ARIMA forecast Utah" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)', marginRight:'2%'}} />
        <img src="/vol3_housing_project/assets/ohio_forecast.png" alt="ARIMA forecast Ohio" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)'}} />

        <h2>VARMAX forecasting on clusters</h2>
        <img src="/vol3_housing_project/assets/utah_varmax.png" alt="VARMAX cluster forecasting" style={{width:'100%', borderRadius:12, border:'1px solid var(--border)'}} />

        <h2>Bayesian hierarchical model</h2>
        <img src="/vol3_housing_project/assets/forest_plot.png" alt="Posterior intercepts by state" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)', marginRight:'2%'}} />
        <img src="/vol3_housing_project/assets/slope_forest.png" alt="Posterior slopes by state" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)'}} />
        <div style={{height:8}} />
        <img src="/vol3_housing_project/assets/prior_predictive_samples.png" alt="Prior predictive distribution" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)', marginRight:'2%'}} />
        <img src="/vol3_housing_project/assets/trace_plot.png" alt="Trace plots" style={{width:'49%', borderRadius:12, border:'1px solid var(--border)'}} />
      </div>
    </section>
  )
}

