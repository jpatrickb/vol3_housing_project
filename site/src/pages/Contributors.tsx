import { Hero } from '../components';

const contributors = [
  { name: 'Patrick Beal', role: 'Lead Analyst' },
  { name: 'Tiara Eddington', role: 'Research Assistant' },
  { name: 'Madelyn Vines', role: 'Research Assistant' },
];

export default function Contributors() {
  return (
    <>
      <Hero
        title="Contributors"
        subtitle="Housing Markets Research Team"
      />
      <section className="overview">
        <div className="container">
          <h2>Our Team</h2>
          <p className="intro-text">
            This project was developed by a dedicated team of researchers working on state-level housing market analysis.
          </p>
          <div className="grid-2">
            {contributors.map((contributor, idx) => (
              <div key={idx} className="card">
                <h3>{contributor.name}</h3>
                <p style={{ color: '#0052cc', fontWeight: 600, marginBottom: 0 }}>
                  {contributor.role}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

