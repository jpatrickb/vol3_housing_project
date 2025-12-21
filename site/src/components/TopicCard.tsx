import React from 'react';

interface TopicCardProps {
  icon: string;
  title: string;
  description: string;
  items?: string[];
}

export const TopicCard: React.FC<TopicCardProps> = ({ icon, title, description, items }) => {
  return (
    <div className="topic-card">
      <div className="topic-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{description}</p>
      {items && items.length > 0 && (
        <ul className="topic-items">
          {items.map((item, idx) => (
            <li key={idx}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  );
};
