const GH = 'https://github.com/jpatrickb/vol3_housing_project/blob/main'

export default function Notebooks() {
  return (
    <section className="container">
      <div className="panel" style={{padding: 24}}>
        <h1 style={{marginTop:0}}>Notebooks</h1>
        <ul>
          <li><a href={`${GH}/notebooks/01_intro.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/01_intro.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/02_data_exploration.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/02_data_exploration.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/02b_county_data.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/02b_county_data.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/03_data_cleaning.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/03_data_cleaning.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/04_interpolation.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/04_interpolation.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/05_detrending.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/05_detrending.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/06_arima.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/06_arima.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/06b_fit_arma.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/06b_fit_arma.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/07_kalman_filter.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/07_kalman_filter.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/08_time_series_clustering.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/08_time_series_clustering.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/09_linear_bayes.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/09_linear_bayes.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/10_bayes_hierarchy.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/10_bayes_hierarchy.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/11_regression.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/11_regression.ipynb</code></a></li>
          <li><a href={`${GH}/notebooks/12_forecasting_with_demographic.ipynb`} target="_blank" rel="noreferrer"><code>notebooks/12_forecasting_with_demographic.ipynb</code></a></li>
        </ul>
      </div>
    </section>
  )
}

