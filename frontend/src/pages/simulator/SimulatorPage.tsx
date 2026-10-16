import { useEffect, useState } from 'react';
import { FileText, Link2 } from 'lucide-react';
import { useAssumptions } from '@/context/AssumptionsContext';
import { VALID } from '@/data/validation';
import { knownThreshold, scale, scenario } from '@/lib/economics';
import { encodeScenario, scenarioLink } from '@/lib/scenario';
import { fmtInt, fmtMultiple, fmtPct, fmtUsd, fmtUsd2, fmtUsdCompact } from '@/lib/format';
import { PageHeader } from '@/components/ui/primitives';
import { CopyButton } from '@/components/ui/CopyButton';
import { CAVEAT_TEXT } from '@/components/ui/CaveatChip';
import { MathVerifiedBadge } from '@/components/ui/MathVerifiedBadge';
import { SimulatorInputs } from '@/pages/simulator/SimulatorInputs';
import { NetValueChart } from '@/pages/simulator/NetValueChart';
import { ResultsPanel } from '@/pages/simulator/ResultsPanel';
import { SensitivityTornado } from '@/pages/simulator/SensitivityTornado';
import { PolicyOptions } from '@/pages/simulator/PolicyOptions';
import { ExecutiveSummary } from '@/pages/simulator/ExecutiveSummary';

export function SimulatorPage() {
  const a = useAssumptions();
  const [reportOpen, setReportOpen] = useState(false);
  const s = scenario(a.k, a);
  const th = knownThreshold(a.k);
  const at = (x: number) => scale(x, a.subscribers);
  const state = { k: a.k, valuePerSave: a.valuePerSave, costPerContact: a.costPerContact, saveRate: a.saveRate, subscribers: a.subscribers };
  const query = encodeScenario(state);
  const link = scenarioLink(state);

  useEffect(() => {
    window.history.replaceState(null, '', `#/simulator?${query}`);
  }, [query]);

  const summary = [
    `Churn retention scenario at ${fmtInt(a.subscribers)} subscribers: contact the top ${fmtPct(s.k / VALID.n, 2)} (${fmtInt(at(s.k))} subs)${th !== undefined ? `, score >= ${th}` : ''} per cycle.`,
    `Assumptions: V=${fmtUsd(a.valuePerSave)}/save, C=${fmtUsd2(a.costPerContact)}/contact, S=${fmtPct(a.saveRate, 0)} save rate.`,
    `Model: reaches ${fmtInt(at(s.tp))} churners (precision ${fmtPct(s.precision)}, recall ${fmtPct(s.recall)}).`,
    `Net ${fmtUsdCompact(at(s.net))}/cycle (${fmtUsdCompact(at(s.net) * 12)}/yr) vs ${fmtUsdCompact(at(s.randomNet))} random; net ROI ${fmtPct(s.netRoi, 0)}, ${fmtMultiple(s.returnOnSpend)} gross return on spend.`,
    `Measured on the ${fmtInt(VALID.n)}-subscriber validation window: net ${fmtUsd(s.net)}/cycle on ${fmtInt(s.k)} contacts.`,
    `Break-even: save rate ${fmtPct(s.breakEvenSaveRate)}, contact cost ${fmtUsd2(s.breakEvenCost)}.`,
    `Caveat: ${CAVEAT_TEXT}`,
    `Open this scenario: ${link}`,
  ].join('\n');

  return (
    <div>
      <PageHeader
        eyebrow="Policy Simulator"
        title="How many subscribers should retention contact?"
        description="Pick how much of the base to contact. Precision and recall come from the champion model's ranking on the time-based validation window; the business assumptions and subscriber base are yours. Dollar figures scale linearly to your base (enter 193,205 to see the measured window)."
        action={
          <div className="flex flex-wrap items-center gap-2">
            <MathVerifiedBadge />
            <CopyButton text={link} label="Copy link" icon={<Link2 className="h-3.5 w-3.5" />} />
            <CopyButton text={summary} label="Copy scenario" />
            <button onClick={() => setReportOpen(true)} className="btn-primary px-3 py-1.5 text-xs">
              <FileText className="h-3.5 w-3.5" /> Executive summary
            </button>
          </div>
        }
      />
      <div className="grid gap-6 lg:grid-cols-12">
        <div className="lg:col-span-4">
          <div className="lg:sticky lg:top-20">
            <SimulatorInputs />
          </div>
        </div>
        <div className="space-y-6 lg:col-span-8">
          <NetValueChart />
          <div className="grid gap-6 xl:grid-cols-2">
            <ResultsPanel s={s} />
            <SensitivityTornado />
          </div>
          <PolicyOptions />
        </div>
      </div>
      {reportOpen && <ExecutiveSummary state={state} link={link} onClose={() => setReportOpen(false)} />}
    </div>
  );
}
