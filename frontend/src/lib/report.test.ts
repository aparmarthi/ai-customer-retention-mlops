import { describe, expect, it } from 'vitest';
import { DEFAULT_SCENARIO } from '@/lib/scenario';
import { VALID } from '@/data/validation';
import { buildExecReport, reportToMarkdown } from '@/lib/report';

const LINK = 'https://example.test/#/simulator';
const DATE = new Date('2026-10-16T12:00:00Z');

describe('buildExecReport', () => {
  it('recommends the default scenario with a positive outcome', () => {
    const r = buildExecReport(DEFAULT_SCENARIO, LINK, DATE);
    expect(r.positive).toBe(true);
    expect(r.headline).toMatch(/^Contact the /);
    expect(r.metrics).toHaveLength(4);
  });

  it('flags a loss-making scenario with break-even guidance', () => {
    const r = buildExecReport({ ...DEFAULT_SCENARIO, costPerContact: 50, saveRate: 0.05 }, LINK, DATE);
    expect(r.positive).toBe(false);
    expect(r.headline).toMatch(/loses/);
    expect(r.metrics[0].tone).toBe('risk');
  });

  it('splits the whole subscriber base across the three bands', () => {
    const subs = 1_000_000;
    const r = buildExecReport({ ...DEFAULT_SCENARIO, subscribers: subs }, LINK, DATE);
    const total = r.bands.reduce((t, b) => t + Number(b.customers.replace(/,/g, '')), 0);
    expect(Math.abs(total - subs)).toBeLessThanOrEqual(2);
    const churners = r.bands.reduce((t, b) => t + Number(b.churners.replace(/,/g, '')), 0);
    expect(Math.abs(churners - (VALID.churners * subs) / VALID.n)).toBeLessThanOrEqual(2);
  });
});

describe('reportToMarkdown', () => {
  it('includes every section and the scenario link', () => {
    const md = reportToMarkdown(buildExecReport(DEFAULT_SCENARIO, LINK, DATE));
    for (const h of ['Recommendation', 'Key numbers', 'Contact plan by risk band', 'Assumptions', 'Model', 'Risks and sensitivities', 'Caveats']) {
      expect(md).toContain(`## ${h}`);
    }
    expect(md).toContain(LINK);
  });
});
