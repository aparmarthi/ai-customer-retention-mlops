import { useEffect, useState } from 'react';
import { ImageOff, Maximize2, X } from 'lucide-react';

interface LightboxProps {
  src: string;
  alt: string;
  onClose: () => void;
}

export function Lightbox({ src, alt, onClose }: LightboxProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={alt}
      className="fixed inset-0 z-[60] flex animate-fade-in items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <button onClick={onClose} autoFocus className="absolute right-4 top-4 rounded-lg bg-zinc-900 p-2 text-zinc-300 hover:text-white" aria-label="Close image">
        <X className="h-5 w-5" />
      </button>
      <img src={src} alt={alt} onClick={(e) => e.stopPropagation()} className="max-h-[90vh] max-w-[95vw] rounded-lg bg-white object-contain shadow-2xl" />
    </div>
  );
}

interface FigureCardProps {
  src: string;
  title: string;
  caption: string;
}

export function FigureCard({ src, title, caption }: FigureCardProps) {
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const [loaded, setLoaded] = useState(false);
  return (
    <figure className="card group overflow-hidden">
      <button
        type="button"
        onClick={() => !failed && setOpen(true)}
        className="relative block aspect-[4/3] w-full overflow-hidden bg-white/95"
        aria-label={`Enlarge ${title}`}
        disabled={failed}
      >
        {!loaded && !failed && <div className="absolute inset-0 animate-pulse bg-zinc-800" />}
        {failed ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 bg-zinc-900 text-zinc-500">
            <ImageOff className="h-6 w-6" />
            <span className="text-xs">Image unavailable</span>
          </div>
        ) : (
          <img
            src={src}
            alt={title}
            loading="lazy"
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
            className="h-full w-full object-contain p-2 transition-transform duration-300 group-hover:scale-[1.02]"
          />
        )}
        {!failed && (
          <span className="absolute right-2 top-2 rounded-md bg-zinc-950/80 p-1.5 text-zinc-300 opacity-0 transition-opacity group-hover:opacity-100">
            <Maximize2 className="h-3.5 w-3.5" />
          </span>
        )}
      </button>
      <figcaption className="border-t border-zinc-800/80 p-4">
        <p className="text-sm font-medium text-zinc-100">{title}</p>
        <p className="mt-1 text-xs text-zinc-500">{caption}</p>
      </figcaption>
      {open && <Lightbox src={src} alt={title} onClose={() => setOpen(false)} />}
    </figure>
  );
}
