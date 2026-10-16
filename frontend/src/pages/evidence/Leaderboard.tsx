import { useMemo, useState } from 'react';
import { ArrowDown, ArrowUp, ArrowUpDown, Crown } from 'lucide-react';
import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import { LEADERBOARD, type LeaderboardRow } from '@/data/leaderboard';
import { Badge, Card } from '@/components/ui/primitives';

type SortKey = 'model' | 'prAuc' | 'rocAuc' | 'f1' | 'precision' | 'recall';

const COLS: Array<{ key: SortKey; label: string; numeric: boolean }> = [
  { key: 'model', label: 'Model', numeric: false },
  { key: 'prAuc', label: 'PR-AUC', numeric: true },
  { key: 'rocAuc', label: 'ROC-AUC', numeric: true },
  { key: 'f1', label: 'F1', numeric: true },
  { key: 'precision', label: 'Precision', numeric: true },
  { key: 'recall', label: 'Recall', numeric: true },
];

const fmt4 = (v: number | undefined) => (v === undefined ? '—' : v.toFixed(4));

export function Leaderboard() {
  const [sort, setSort] = useState<{ key: SortKey; dir: 'asc' | 'desc' }>({ key: 'prAuc', dir: 'desc' });

  const rows = useMemo(() => {
    const val = (r: LeaderboardRow) => r[sort.key];
    return [...LEADERBOARD].sort((a, b) => {
      const x = val(a);
      const y = val(b);
      if (x === undefined) return 1;
      if (y === undefined) return -1;
      const c = typeof x === 'string' ? x.localeCompare(String(y)) : (x as number) - (y as number);
      return sort.dir === 'asc' ? c : -c;
    });
  }, [sort]);

  const toggle = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'model' ? 'asc' : 'desc' }));

  const chartData = [...LEADERBOARD].sort((a, b) => b.prAuc - a.prAuc);

  return (
    <Card
      title="Model-family benchmark (random split)"
      subtitle="Seven model families on a random 80/20 split at 6.4% prevalence. Baselines, ensemble and AutoML runs (12 experiments in total) are in leaderboard.md."
      action={<Badge tone="info">Random split</Badge>}
      bodyClassName="p-0"
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left">
              {COLS.map((c) => {
                const active = sort.key === c.key;
                const Icon = !active ? ArrowUpDown : sort.dir === 'asc' ? ArrowUp : ArrowDown;
                return (
                  <th
                    key={c.key}
                    scope="col"
                    aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
                    className={`px-4 py-3 ${c.numeric ? 'text-right' : ''}`}
                  >
                    <button onClick={() => toggle(c.key)} className={`inline-flex items-center gap-1 text-xs font-medium ${active ? 'text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'}`}>
                      {c.label}
                      <Icon className="h-3 w-3" />
                    </button>
                  </th>
                );
              })}
              <th scope="col" className="px-4 py-3 text-right text-xs font-medium text-zinc-500">Training</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className={`border-b border-zinc-800/60 transition-colors last:border-0 ${r.champion ? 'bg-emerald-500/[0.07]' : 'hover:bg-zinc-800/30'}`}>
                <td className="px-4 py-3">
                  <span className="flex items-center gap-2 font-medium text-zinc-100">
                    {r.champion && <Crown className="h-3.5 w-3.5 text-emerald-400" aria-label="Champion" />}
                    {r.model}
                    {r.champion && <Badge tone="value">Champion</Badge>}
                  </span>
                </td>
                {COLS.slice(1).map((c) => (
                  <td key={c.key} className={`px-4 py-3 text-right tabular-nums ${r.champion && c.key === 'prAuc' ? 'font-semibold text-emerald-300' : 'text-zinc-300'}`}>
                    {fmt4(r[c.key] as number | undefined)}
                  </td>
                ))}
                <td className="whitespace-nowrap px-4 py-3 text-right text-xs text-zinc-500">{r.train}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="border-t border-zinc-800 p-5">
        <p className="label mb-3">PR-AUC by model</p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 48, top: 0, bottom: 0 }}>
              <XAxis type="number" domain={[0, 1]} hide />
              <YAxis type="category" dataKey="model" width={110} tick={{ fill: '#a1a1aa', fontSize: 12 }} axisLine={false} tickLine={false} />
              <Bar dataKey="prAuc" radius={[0, 4, 4, 0]} barSize={18}>
                {chartData.map((d) => (
                  <Cell key={d.model} fill={d.champion ? '#34d399' : '#52525b'} />
                ))}
                <LabelList dataKey="prAuc" position="right" formatter={(v: number) => v.toFixed(3)} fill="#d4d4d8" fontSize={11} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  );
}
