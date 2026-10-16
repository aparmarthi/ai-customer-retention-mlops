import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, type TooltipProps } from 'recharts';
import { useAssumptions } from '@/context/AssumptionsContext';
import { CAPACITY_K, ROI_MAX_K } from '@/data/validation';
import { logSpace, net, randomNet } from '@/lib/economics';
import { fmtCompact, fmtInt, fmtUsd, fmtUsdCompact } from '@/lib/format';
import { Card } from '@/components/ui/primitives';

interface Point {
  k: number;
  model: number;
  random: number;
}

function ChartTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload as Point;
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900/95 p-3 text-xs shadow-xl tabular-nums">
      <p className="font-medium text-zinc-100">K = {fmtInt(p.k)}</p>
      <p className="mt-1 text-emerald-300">Model: {fmtUsd(p.model)}</p>
      <p className="text-rose-300">Random: {fmtUsd(p.random)}</p>
      <p className="mt-1 text-zinc-500">Click or drag to set K</p>
    </div>
  );
}

export function NetValueChart() {
  const { k, setK, valuePerSave: V, costPerContact: C, saveRate: S } = useAssumptions();
  const [dragging, setDragging] = useState(false);

  const data = useMemo<Point[]>(() => {
    const ks = Array.from(new Set([...logSpace(100, 50000, 120), 1262, 1478, 5000, 10000, 20000])).sort((a, b) => a - b);
    return ks.map((kk) => ({ k: kk, model: net(kk, V, C, S), random: randomNet(kk, V, C, S) }));
  }, [V, C, S]);

  const pick = (label: unknown) => {
    const v = Number(label);
    if (Number.isFinite(v) && v > 0) setK(v);
  };

  return (
    <Card title="Net value vs contacts" subtitle="Per cycle, at validation scale. Model ranking (solid) vs random targeting (dashed).">
      <div className="h-80 select-none" role="img" aria-label={`Net value chart. Current K is ${fmtInt(k)}.`}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 24, right: 16, left: 4, bottom: 0 }}
            onMouseDown={(s) => {
              setDragging(true);
              pick(s?.activeLabel);
            }}
            onMouseMove={(s) => dragging && pick(s?.activeLabel)}
            onMouseUp={() => setDragging(false)}
            onMouseLeave={() => setDragging(false)}
            style={{ cursor: 'crosshair' }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="k"
              type="number"
              scale="log"
              domain={[100, 50000]}
              ticks={[100, 300, 1000, 3000, 10000, 30000, 50000]}
              tickFormatter={(v: number) => fmtCompact(v)}
              tick={{ fill: '#71717a', fontSize: 11 }}
              axisLine={{ stroke: '#3f3f46' }}
              tickLine={false}
            />
            <YAxis tickFormatter={(v: number) => fmtUsdCompact(v)} tick={{ fill: '#71717a', fontSize: 11 }} axisLine={false} tickLine={false} width={64} />
            <ReferenceLine y={0} stroke="#52525b" />
            <ReferenceLine x={ROI_MAX_K} stroke="#a1a1aa" strokeDasharray="2 4" label={{ value: 'ROI-max (K=1,478)', position: 'insideTopLeft', fill: '#a1a1aa', fontSize: 10, dy: -20 }} />
            <ReferenceLine x={CAPACITY_K} stroke="#a1a1aa" strokeDasharray="2 4" label={{ value: 'Capacity policy (K=10,000)', position: 'insideTopRight', fill: '#a1a1aa', fontSize: 10, dy: -20 }} />
            <ReferenceLine x={k} stroke="#34d399" strokeWidth={1.5} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: '#e4e4e7', strokeDasharray: '3 3' }} />
            <Legend verticalAlign="bottom" height={24} iconType="plainline" wrapperStyle={{ fontSize: 12, color: '#a1a1aa' }} />
            <Line name="Model" type="monotone" dataKey="model" stroke="#34d399" strokeWidth={2.5} dot={false} isAnimationActive={false} />
            <Line name="Random" type="monotone" dataKey="random" stroke="#fb7185" strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
