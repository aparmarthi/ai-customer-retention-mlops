import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';

type ToastKind = 'error' | 'success' | 'info';

interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  detail?: string;
}

interface ToastContextValue {
  notify: (kind: ToastKind, title: string, detail?: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const styles: Record<ToastKind, { icon: typeof Info; ring: string; color: string }> = {
  error: { icon: AlertCircle, ring: 'border-rose-500/40', color: 'text-rose-400' },
  success: { icon: CheckCircle2, ring: 'border-emerald-500/40', color: 'text-emerald-400' },
  info: { icon: Info, ring: 'border-sky-500/40', color: 'text-sky-400' },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const notify = useCallback(
    (kind: ToastKind, title: string, detail?: string) => {
      const id = nextId.current++;
      setToasts((t) => [...t.slice(-3), { id, kind, title, detail }]);
      setTimeout(() => dismiss(id), kind === 'error' ? 8000 : 4000);
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={{ notify }}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[70] flex w-[min(92vw,380px)] flex-col gap-2" aria-live="polite">
        {toasts.map((t) => {
          const s = styles[t.kind];
          const Icon = s.icon;
          return (
            <div
              key={t.id}
              role={t.kind === 'error' ? 'alert' : 'status'}
              className={`pointer-events-auto flex animate-slide-up items-start gap-3 rounded-xl border bg-zinc-900/95 p-3 shadow-2xl shadow-black/50 backdrop-blur ${s.ring}`}
            >
              <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${s.color}`} aria-hidden />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-zinc-100">{t.title}</p>
                {t.detail && <p className="mt-0.5 break-words text-xs text-zinc-400">{t.detail}</p>}
              </div>
              <button onClick={() => dismiss(t.id)} className="rounded p-0.5 text-zinc-500 hover:text-zinc-200" aria-label="Dismiss notification">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside ToastProvider');
  return ctx;
}
