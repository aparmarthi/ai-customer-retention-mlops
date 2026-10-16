import { CAPACITY_THRESHOLD, DEFAULT_THRESHOLD } from '@/data/validation';
import { RISK_BANDS, riskBand } from '@/pages/score/risk';

const R = 80;
const CX = 100;
const CY = 96;

function polar(p: number) {
  const a = Math.PI * (1 - p);
  return { x: CX + R * Math.cos(a), y: CY - R * Math.sin(a) };
}

function arc(p0: number, p1: number) {
  const a = polar(p0);
  const b = polar(p1);
  return `M ${a.x} ${a.y} A ${R} ${R} 0 0 1 ${b.x} ${b.y}`;
}

export function Gauge({ value }: { value: number }) {
  const band = RISK_BANDS[riskBand(value)];
  const needle = polar(Math.min(1, Math.max(0, value)));
  return (
    <svg viewBox="0 0 200 112" className="w-full max-w-xs" role="img" aria-label={`Churn probability ${(value * 100).toFixed(1)} percent, ${band.label}`}>
      <path d={arc(0, CAPACITY_THRESHOLD)} stroke="#34d399" strokeOpacity={0.25} strokeWidth={14} fill="none" />
      <path d={arc(CAPACITY_THRESHOLD, DEFAULT_THRESHOLD)} stroke="#fbbf24" strokeOpacity={0.25} strokeWidth={14} fill="none" />
      <path d={arc(DEFAULT_THRESHOLD, 1)} stroke="#fb7185" strokeOpacity={0.25} strokeWidth={14} fill="none" />
      <path d={arc(0, Math.max(0.001, value))} stroke={band.hex} strokeWidth={14} fill="none" strokeLinecap="round" style={{ transition: 'all 600ms ease' }} />
      <line x1={CX} y1={CY} x2={needle.x} y2={needle.y} stroke="#e4e4e7" strokeWidth={2} strokeLinecap="round" />
      <circle cx={CX} cy={CY} r={4} fill="#e4e4e7" />
      <text x={CX} y={CY - 26} textAnchor="middle" className="fill-zinc-50 text-[26px] font-semibold tabular-nums">
        {(value * 100).toFixed(1)}%
      </text>
      <text x={20} y={110} textAnchor="middle" className="fill-zinc-600 text-[9px]">0</text>
      <text x={180} y={110} textAnchor="middle" className="fill-zinc-600 text-[9px]">1</text>
    </svg>
  );
}
