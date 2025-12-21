import React from 'react';

interface Finding {
  label: string;
  text: string;
}

interface FindingCardProps {
  icon: string;
  title: string;
  findings: Finding[];
}

export const FindingCard: React.FC<FindingCardProps> = ({ icon, title, findings }) => {
  return (
    <div className="finding-card">
      <div className="finding-icon">{icon}</div>
      <h3>{title}</h3>
      <p>
        {findings.map((finding, idx) => (
          <React.Fragment key={idx}>
            <strong>{finding.label}:</strong> {finding.text}
            {idx < findings.length - 1 && <br />}
          </React.Fragment>
        ))}
      </p>
    </div>
  );
};
