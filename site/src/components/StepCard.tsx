import React from 'react';

interface StepCardProps {
  number: number;
  title: string;
  description: string;
  children?: React.ReactNode;
}

export const StepCard: React.FC<StepCardProps> = ({ number, title, description, children }) => {
  return (
    <div className="practice">
      <div className="practice-number">{number}</div>
      <div className="practice-content">
        <h4>{title}</h4>
        <p>{description}</p>
        {children}
      </div>
    </div>
  );
};
