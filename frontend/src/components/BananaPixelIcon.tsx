import { memo } from 'react';

interface BananaPixelIconProps {
  className?: string;
  size?: number;
}

/**
 * Simple pixel-art banana rendered with SVG rectangles so it can scale nicely.
 */
const BananaPixelIconComponent = ({ className = '', size = 64 }: BananaPixelIconProps) => {
  return (
    <svg
      role="img"
      aria-label="Icône pixel art de banane"
      width={size}
      height={size}
      viewBox="0 0 32 32"
      className={className}
    >
      <defs>
        <linearGradient id="bananaShine" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#fff3b0" />
          <stop offset="50%" stopColor="#ffdd57" />
          <stop offset="100%" stopColor="#f0b429" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" fill="none" />
      <g fill="#2f1e0c">
        <rect x="6" y="5" width="2" height="2" />
        <rect x="7" y="6" width="2" height="2" />
        <rect x="8" y="7" width="2" height="2" />
        <rect x="9" y="8" width="2" height="2" />
      </g>
      <g fill="url(#bananaShine)">
        <rect x="9" y="9" width="2" height="2" />
        <rect x="10" y="10" width="2" height="2" />
        <rect x="11" y="11" width="2" height="2" />
        <rect x="12" y="12" width="2" height="2" />
        <rect x="13" y="13" width="2" height="2" />
        <rect x="14" y="14" width="2" height="2" />
        <rect x="15" y="15" width="2" height="2" />
        <rect x="16" y="16" width="2" height="2" />
        <rect x="17" y="17" width="2" height="2" />
        <rect x="18" y="17" width="2" height="2" />
        <rect x="19" y="16" width="2" height="2" />
        <rect x="20" y="15" width="2" height="2" />
        <rect x="21" y="14" width="2" height="2" />
        <rect x="22" y="13" width="2" height="2" />
        <rect x="23" y="12" width="2" height="2" />
        <rect x="24" y="11" width="2" height="2" />
        <rect x="25" y="10" width="2" height="2" />
        <rect x="26" y="9" width="2" height="2" />
        <rect x="24" y="18" width="2" height="2" />
        <rect x="23" y="19" width="2" height="2" />
        <rect x="22" y="20" width="2" height="2" />
        <rect x="21" y="21" width="2" height="2" />
        <rect x="20" y="22" width="2" height="2" />
        <rect x="19" y="23" width="2" height="2" />
        <rect x="18" y="24" width="2" height="2" />
        <rect x="17" y="25" width="2" height="2" />
        <rect x="16" y="26" width="2" height="2" />
      </g>
      <g fill="#f7b731">
        <rect x="20" y="18" width="2" height="2" />
        <rect x="19" y="19" width="2" height="2" />
        <rect x="18" y="20" width="2" height="2" />
        <rect x="17" y="21" width="2" height="2" />
        <rect x="16" y="22" width="2" height="2" />
        <rect x="15" y="23" width="2" height="2" />
        <rect x="14" y="24" width="2" height="2" />
      </g>
      <g fill="#d17b0f">
        <rect x="24" y="9" width="2" height="2" />
        <rect x="25" y="8" width="2" height="2" />
        <rect x="26" y="7" width="2" height="2" />
      </g>
    </svg>
  );
};

export const BananaPixelIcon = memo(BananaPixelIconComponent);
