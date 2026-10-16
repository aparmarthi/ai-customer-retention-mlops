import { Layers } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { CAPACITY_K, DEFAULT_THRESHOLD, ROI_MAX_K, VALID } from '@/data/validation';
import { TIERS, net, scale, tieredNet, tpAtK } from '@/lib/economics';
import { fmtInt, fmtPct, fmtUsd2, fmtUsdCompact } from '@/lib/format';
import { Badge, Card } from '@/components/ui/primitives';

interface Option {
  name: string;
  detail: string;
  contacts: number;
  churners: number;
  spend: number;
  net: number;
}

export function PolicyOptions() {
  const { valuePerSave: V, costPerContact: C, saveRate: S, subscribers, setK } = useAssumptions();
  const at = (x: number) => scale(x, subscribers);
  const t = tieredNet(V, C, S);

  const options: Option[] = [
    {
      name: 'ROI-max threshold',
      detail: `Score ≥ ${DEFAULT_THRESHOLD} (top ${fmtPct(ROI_MAX_K / VALID.n, 2)}), ${fmtUsd2(C)} incentive. API default.`,
      contacts: ROI_MAX_K,
      churners: tpAtK(ROI_MAX_K),
      spend: ROI_MAX_K * C,
      net: net(ROI_MAX_K, V, C, S),
    },
    {
      name: 'Fixed capacity, one channel',
      detail: `Top ${fmtPct(CAPACITY_K / VALID.n, 1)} (${fmtInt(CAPACITY_K)} on the validation window); everyone gets the ${fmtUsd2(C)} incentive.`,
      contacts: CAPACITY_K,
      churners: tpAtK(CAPACITY_K),
      spend: CAPACITY_K * C,
      net: net(CAPACITY_K, V, C, S),
    },
    {
      name: 'Tiered',
      detail: `${fmtUsd2(C)} incentive for the top ${fmtPct(TIERS.tier1.k / VALID.n, 2)}; ${fmtUsd2(TIERS.tier2.cost)} email or in-app nudge at a ${fmtPct(TIERS.tier2.saveRate, 0)} save rate for the rest of the top ${fmtPct(TIERS.tier2.kEnd / VALID.n, 1)}.`,
      contacts: TIERS.tier2.kEnd,
      churners: tpAtK(TIERS.tier2.kEnd),
      spend: t.spend,
      net: t.combined,
    },
  ];
  const best = Math.max(...options.map((o) => o.net));

  return (
    <Card
      title="Which policy should we run?"
      subtitle="The three policies the API and risk bands support, at your assumptions and subscriber base, per cycle."
      icon={<Layers className="h-4 w-4" />}
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-sm">
          <thead className="text-left text-[11px] uppercase tracking-wider text-zinc-500">
            <tr className="border-b border-zinc-800">
              <th scope="col" className="py-2 pr-3 font-medium">Policy</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Contacts</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Churners reached</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Spend</th>
              <th scope="col" className="py-2 text-right font-medium">Net / cycle</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/80">
            {options.map((o) => (
              <tr key={o.name} className="align-top">
                <td className="py-3 pr-3">
                  <span className="flex items-center gap-2 font-medium text-zinc-100">
                    {o.name}
                    {o.net === best && best > 0 && <Badge tone="value">Best here</Badge>}
                  </span>
                  <span className="mt-0.5 block text-[11px] text-zinc-500">{o.detail}</span>
                </td>
                <td className="py-3 pr-3 text-right tabular-nums text-zinc-200">{fmtInt(at(o.contacts))}</td>
                <td className="py-3 pr-3 text-right tabular-nums text-zinc-200">{fmtInt(at(o.churners))}</td>
                <td className="py-3 pr-3 text-right tabular-nums text-zinc-200">{fmtUsdCompact(at(o.spend))}</td>
                <td className={`py-3 text-right font-semibold tabular-nums ${o.net >= 0 ? 'text-emerald-300' : 'text-rose-300'}`}>{fmtUsdCompact(at(o.net))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Badge tone="warn">Tier-2 nudge cost and save rate are illustrative assumptions</Badge>
        <button onClick={() => setK(ROI_MAX_K)} className="btn-ghost px-2 py-1 text-xs">Set K = {fmtInt(ROI_MAX_K)}</button>
        <button onClick={() => setK(CAPACITY_K)} className="btn-ghost px-2 py-1 text-xs">Set K = {fmtInt(CAPACITY_K)}</button>
      </div>
      <p className="mt-3 text-[11px] text-zinc-500">
        At the default assumptions ($120 per save, $5 per contact, 20% save rate), giving the incentive to the whole top 10,000 loses money. Fixed capacity pays only when everyone below the top 0.76% gets a cheap channel, which is why policy v2 gives that band a nudge.
      </p>
    </Card>
  );
}
