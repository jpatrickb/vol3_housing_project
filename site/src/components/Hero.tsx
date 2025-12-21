import React from 'react';

interface HeroButton {
  label: string;
  href: string;
  icon?: string;
  variant: 'primary' | 'secondary';
}

interface HeroProps {
  title: string;
  subtitle?: string;
  author?: string;
  buttons?: HeroButton[];
}

export const Hero: React.FC<HeroProps> = ({ title, subtitle, author, buttons }) => {
  return (
    <section className="hero">
      <div className="container">
        <div className="hero-content">
          <h1>{title}</h1>
          {subtitle && <p className="subtitle">{subtitle}</p>}
          {author && <p className="author">{author}</p>}
          {buttons && buttons.length > 0 && (
            <div className="cta-buttons">
              {buttons.map((btn, idx) => (
                <a
                  key={idx}
                  href={btn.href}
                  className={`btn btn-${btn.variant}`}
                >
                  {btn.icon && <span className="icon">{btn.icon}</span>}
                  {btn.label}
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
