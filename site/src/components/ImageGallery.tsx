import React from 'react';

interface Image {
  src: string;
  alt: string;
  caption?: string;
}

interface ImageGalleryProps {
  images: Image[];
  columns?: 1 | 2;
}

export const ImageGallery: React.FC<ImageGalleryProps> = ({ images, columns = 2 }) => {
  const gridClass = columns === 1 ? 'image-grid-full' : 'image-grid-2';

  return (
    <div className={gridClass}>
      {images.map((img, idx) => (
        <div key={idx}>
          <img src={img.src} alt={img.alt} className="result-image" />
          {img.caption && <p className="image-caption">{img.caption}</p>}
        </div>
      ))}
    </div>
  );
};
