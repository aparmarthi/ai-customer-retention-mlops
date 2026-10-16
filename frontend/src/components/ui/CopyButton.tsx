import { useState, type ReactNode } from 'react';
import { Check, Copy } from 'lucide-react';
import { useToast } from '@/context/ToastContext';

export function CopyButton({ text, label = 'Copy', className = '', icon }: { text: string; label?: string; className?: string; icon?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const { notify } = useToast();
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      notify('error', 'Clipboard unavailable', 'Your browser blocked clipboard access.');
    }
  };
  return (
    <button type="button" onClick={copy} className={`btn-secondary px-2.5 py-1.5 text-xs ${className}`} aria-label={label}>
      {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : icon ?? <Copy className="h-3.5 w-3.5" />}
      {copied ? 'Copied' : label}
    </button>
  );
}

export function CodeBlock({ code, copyLabel = 'Copy' }: { code: string; copyLabel?: string }) {
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950 p-4 pr-24 font-mono text-xs leading-relaxed text-zinc-300">
        {code}
      </pre>
      <CopyButton text={code} label={copyLabel} className="absolute right-2 top-2" />
    </div>
  );
}
