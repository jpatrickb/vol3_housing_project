import { Hero, TopicCard } from '../components';

const GH = 'https://github.com/jpatrickb/vol3_housing_project/blob/main';

interface NotebookGroup {
  icon: string;
  title: string;
  description: string;
  notebooks: { name: string; url: string }[];
}

const notebookGroups: NotebookGroup[] = [
  {
    icon: '📊',
    title: 'Data Preparation',
    description: 'Data loading, exploration, cleaning, and preprocessing',
    notebooks: [
      { name: '01_intro.ipynb', url: `${GH}/notebooks/01_intro.ipynb` },
      { name: '02_data_exploration.ipynb', url: `${GH}/notebooks/02_data_exploration.ipynb` },
      { name: '02b_county_data.ipynb', url: `${GH}/notebooks/02b_county_data.ipynb` },
      { name: '03_data_cleaning.ipynb', url: `${GH}/notebooks/03_data_cleaning.ipynb` },
      { name: '04_interpolation.ipynb', url: `${GH}/notebooks/04_interpolation.ipynb` },
    ],
  },
  {
    icon: '📈',
    title: 'Time-Series Analysis',
    description: 'Detrending, ARIMA models, and Kalman filtering',
    notebooks: [
      { name: '05_detrending.ipynb', url: `${GH}/notebooks/05_detrending.ipynb` },
      { name: '06_arima.ipynb', url: `${GH}/notebooks/06_arima.ipynb` },
      { name: '06b_fit_arma.ipynb', url: `${GH}/notebooks/06b_fit_arma.ipynb` },
      { name: '07_kalman_filter.ipynb', url: `${GH}/notebooks/07_kalman_filter.ipynb` },
    ],
  },
  {
    icon: '🔬',
    title: 'Clustering & Bayesian',
    description: 'Time-series clustering and Bayesian hierarchical models',
    notebooks: [
      { name: '08_time_series_clustering.ipynb', url: `${GH}/notebooks/08_time_series_clustering.ipynb` },
      { name: '09_linear_bayes.ipynb', url: `${GH}/notebooks/09_linear_bayes.ipynb` },
      { name: '10_bayes_hierarchy.ipynb', url: `${GH}/notebooks/10_bayes_hierarchy.ipynb` },
      { name: '11_regression.ipynb', url: `${GH}/notebooks/11_regression.ipynb` },
    ],
  },
  {
    icon: '🎯',
    title: 'Forecasting',
    description: 'Forecasting models with demographic features',
    notebooks: [
      { name: '12_forecasting_with_demographic.ipynb', url: `${GH}/notebooks/12_forecasting_with_demographic.ipynb` },
    ],
  },
];

export default function Notebooks() {
  return (
    <>
      <Hero
        title="Jupyter Notebooks"
        subtitle="Complete Analysis Pipeline"
      />
      <section className="topics-section">
        <div className="container">
          <h2>Notebooks by Category</h2>
          <p className="intro-text">
            Explore our complete analysis through interactive Jupyter notebooks. Each notebook is available on GitHub for full transparency and reproducibility.
          </p>
          <div className="topics-grid">
            {notebookGroups.map((group, idx) => (
              <TopicCard
                key={idx}
                icon={group.icon}
                title={group.title}
                description={group.description}
                items={group.notebooks.map((nb) => (
                  <a key={nb.name} href={nb.url} target="_blank" rel="noreferrer" style={{ textDecoration: 'underline', color: '#0052cc' }}>
                    {nb.name}
                  </a>
                ))}
              />
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

