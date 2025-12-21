import React from 'react';

interface SectionProps {
  variant?: 'default' | 'findings' | 'concepts' | 'download' | 'practices' | 'takeaways' | 'topics';
  className?: string;
  children: React.ReactNode;
}

export const Section: React.FC<SectionProps> = ({ variant = 'default', className = '', children }) => {
  const sectionClass = variant === 'default' ? '' : variant;
  const combinedClass = `${sectionClass} ${className}`.trim();

  return (
    <section className={combinedClass}>
      <div className="container">{children}</div>
    </section>
  );
};
