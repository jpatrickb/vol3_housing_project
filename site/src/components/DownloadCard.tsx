import React from 'react';

interface DownloadCardProps {
  icon: string;
  title: string;
  description: string;
  href: string;
  ctaText?: string;
}

export const DownloadCard: React.FC<DownloadCardProps> = ({
  icon,
  title,
  description,
  href,
  ctaText = 'Read PDF →',
}) => {
  return (
    <a href={href} className="download-card" target="_blank" rel="noopener noreferrer">
      <div className="download-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>
      <span className="download-cta">{ctaText}</span>
    </a>
  );
};
