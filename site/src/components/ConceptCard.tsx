import React from 'react';

interface ConceptCardProps {
  title: string;
  definition: string;
}

export const ConceptCard: React.FC<ConceptCardProps> = ({ title, definition }) => {
  return (
    <div className="concept">
      <h4>{title}</h4>
      <p>{definition}</p>
    </div>
  );
};
