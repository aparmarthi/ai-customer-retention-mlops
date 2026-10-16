import { CAPACITY_K, CAPACITY_THRESHOLD, DEFAULT_THRESHOLD, ROI_MAX_K, VALID } from '@/data/validation';
import { joinDrivers, knownThreshold, scale, scenario, sensitivity, tpAtK } from '@/lib/economics';
import type { ScenarioState } from '@/lib/scenario';
import { fmtInt, fmtMultiple, fmtPct, fmtUsd2, fmtUsdCompact } from '@/lib/format';
import { CAVEAT_TEXT } from '@/components/ui/CaveatChip';
import { RISK_BANDS } from '@/pages/score/risk';

export interface ReportItem {
  label: string;
  value: string;
  note?: string;
  tone?: 'value' | 'risk';
}

export interface ReportBand {
  label: string;
  criterion: string;
  customers: string;
  share: string;
  churners: string;
  action: string;
}

export interface ExecReport {
  title: string;
  generatedOn: string;
  positive: boolean;
  headline: string;
  metrics: ReportItem[];
  bands: ReportBand[];
  assumptions: ReportItem[];
  model: ReportItem[];
  risks: string[];
  caveats: string[];
  link: string;
}

export function buildExecReport(state: ScenarioState, link: string, now: Date = new Date()): ExecReport {
  const { k, valuePerSave: V, costPerContact: C, saveRate: S, subscribers: subs } = state;
  const s = scenario(k, state);
  const at = (x: number) => scale(x, subs);
  const contacts = at(k);
  const net = at(s.net);
  const uplift = at(s.net - s.randomNet);
  const positive = s.net >= 0;
  const breakEvenV = s.tp > 0 ? (k * C) / (s.tp * S) : Infinity;
  const th = knownThreshold(k);

  const headline = positive
    ? `Contact the ${fmtInt(contacts)} highest-risk subscribers each cycle (top ${fmtPct(k / VALID.n, 2)} of the base). ` +
      `Expected net value is ${fmtUsdCompact(net)} per cycle (${fmtUsdCompact(net * 12)} a year), ${fmtUsd2(s.returnOnSpend)} of retained value for every $1 of contact spend ` +
      `and ${fmtUsdCompact(uplift)} more per cycle than contacting the same number of subscribers at random.`
    : `At these assumptions, contacting the top ${fmtInt(contacts)} subscribers loses ${fmtUsdCompact(-net)} per cycle. ` +
      `To break even, each save must be worth more than ${Number.isFinite(breakEvenV) ? fmtUsd2(breakEvenV) : 'the current value'}, ` +
      `or the save rate must exceed ${fmtPct(s.breakEvenSaveRate)}.`;

  const bandRow = (label: string, criterion: string, from: number, to: number, action: string): ReportBand => ({
    label,
    criterion,
    customers: fmtInt(at(to - from)),
    share: fmtPct((to - from) / VALID.n, 1),
    churners: fmtInt(at(tpAtK(to) - tpAtK(from))),
    action,
  });

  const { bars, leaders } = sensitivity(k, state);
  const top = bars[0];

  return {
    title: 'Churn retention plan',
    generatedOn: now.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
    positive,
    headline,
    metrics: [
      { label: 'Expected net value / cycle', value: fmtUsdCompact(net), note: `${fmtUsdCompact(net * 12)} a year`, tone: positive ? 'value' : 'risk' },
      { label: 'Net ROI', value: fmtPct(s.netRoi, 0), note: `${fmtMultiple(s.returnOnSpend)} gross return on ${fmtUsdCompact(at(s.spend))} spend / cycle` },
      { label: 'Churners reached', value: fmtPct(s.recall, 0), note: `${fmtInt(at(s.tp))} of ${fmtInt(at(VALID.churners))} at risk` },
      { label: 'Hit rate of contacts', value: fmtPct(s.precision, 0), note: `${fmtMultiple(s.precision / VALID.churnRate, 0)} better than random` },
    ],
    bands: [
      bandRow(RISK_BANDS.contact.label, `Score \u2265 ${DEFAULT_THRESHOLD}`, 0, ROI_MAX_K, RISK_BANDS.contact.action),
      bandRow(RISK_BANDS.capacity.label, `${CAPACITY_THRESHOLD}\u2013${DEFAULT_THRESHOLD}`, ROI_MAX_K, CAPACITY_K, RISK_BANDS.capacity.action),
      bandRow(RISK_BANDS.monitor.label, `Score < ${CAPACITY_THRESHOLD}`, CAPACITY_K, VALID.n, RISK_BANDS.monitor.action),
    ],
    assumptions: [
      { label: 'Subscribers', value: fmtInt(subs) },
      { label: 'Contacts per cycle (K)', value: fmtInt(contacts), note: `${fmtInt(k)} at validation scale${th !== undefined ? `, score \u2265 ${th}` : ''}` },
      { label: 'Value per save', value: fmtUsd2(V) },
      { label: 'Cost per contact', value: fmtUsd2(C) },
      { label: 'Save rate', value: fmtPct(S, 0) },
    ],
    model: [
      { label: 'Champion model', value: VALID.champion },
      { label: 'Validation window', value: `${VALID.window}, ${fmtInt(VALID.n)} subscribers` },
      { label: 'Ranking quality', value: `ROC AUC ${VALID.rocAuc} \u00b7 PR AUC ${VALID.prAuc}` },
      { label: 'Base churn rate', value: fmtPct(VALID.churnRate, 2) },
    ],
    risks: [
      `${leaders.length > 1 ? `${joinDrivers(leaders)} are tied as the most sensitive assumptions` : `${top.driver} is the most sensitive assumption`}: a \u00b150% change in ${leaders.length > 1 ? 'either' : 'it'} moves net value from ${fmtUsdCompact(net + at(Math.min(top.low, top.high)))} to ${fmtUsdCompact(net + at(Math.max(top.low, top.high)))} per cycle.`,
      `Break-even points: save rate ${fmtPct(s.breakEvenSaveRate)} or cost per contact ${fmtUsd2(s.breakEvenCost)}.`,
      `The ${fmtPct(S, 0)} save rate is assumed, not measured. Run a holdout experiment before scaling the programme.`,
    ],
    caveats: [
      CAVEAT_TEXT,
      `Scaling assumes your base has the same churn rate (${fmtPct(VALID.churnRate, 2)}) and risk mix as the validation window.`,
      'Customer-level reasons shown in the app are indicative rule-of-thumb drivers, not the model\u2019s own attributions.',
    ],
    link,
  };
}

const mdTable = (headers: string[], rows: string[][]) =>
  [`| ${headers.join(' | ')} |`, `| ${headers.map(() => '---').join(' | ')} |`, ...rows.map((r) => `| ${r.join(' | ')} |`)].join('\n');

const withNote = (i: ReportItem) => (i.note ? `${i.value} (${i.note})` : i.value);

export function reportToMarkdown(r: ExecReport): string {
  return [
    `# ${r.title}`,
    `_Executive summary \u00b7 ${r.generatedOn}_`,
    `## Recommendation`,
    r.headline,
    `## Key numbers`,
    mdTable(['Metric', 'Value'], r.metrics.map((m) => [m.label, withNote(m)])),
    `## Contact plan by risk band`,
    mdTable(['Band', 'Criterion', 'Subscribers', 'Share', 'Expected churners', 'Action'], r.bands.map((b) => [b.label, b.criterion, b.customers, b.share, b.churners, b.action])),
    `## Assumptions`,
    mdTable(['Assumption', 'Value'], r.assumptions.map((a) => [a.label, withNote(a)])),
    `## Model`,
    mdTable(['Item', 'Value'], r.model.map((m) => [m.label, m.value])),
    `## Risks and sensitivities`,
    r.risks.map((x) => `- ${x}`).join('\n'),
    `## Caveats`,
    r.caveats.map((x) => `- ${x}`).join('\n'),
    `Open this scenario: ${r.link}`,
  ].join('\n\n');
}
