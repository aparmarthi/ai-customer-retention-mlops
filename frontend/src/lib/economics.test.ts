import { describe, expect, it } from 'vitest';
import { runSanityChecks } from '@/lib/sanity';
import { logSpace, net, sensitivity, tieredNet, tpAtK } from '@/lib/economics';
import { CURVE } from '@/data/validation';

describe('documented sanity values', () => {
  for (const c of runSanityChecks()) {
    it(c.label, () => {
      expect(Math.abs(c.actual - c.expected)).toBeLessThanOrEqual(c.tolerance);
    });
  }
});

describe('tpAtK interpolation', () => {
  it('returns 0 for non-positive K', () => {
    expect(tpAtK(0)).toBe(0);
    expect(tpAtK(-5)).toBe(0);
  });

  it('hits every curve point exactly', () => {
    for (const p of CURVE) expect(tpAtK(p.k)).toBeCloseTo(p.tp, 9);
  });

  it('is monotonic non-decreasing', () => {
    let prev = 0;
    for (let k = 1; k <= CURVE[CURVE.length - 1].k + 1000; k += 97) {
      const tp = tpAtK(k);
      expect(tp).toBeGreaterThanOrEqual(prev);
      prev = tp;
    }
  });

  it('caps at the last curve point', () => {
    const last = CURVE[CURVE.length - 1];
    expect(tpAtK(last.k * 10)).toBe(last.tp);
  });
});

describe('sensitivity', () => {
  it('sorts bars by swing size, largest first', () => {
    const { bars } = sensitivity(1478, { valuePerSave: 120, costPerContact: 5, saveRate: 0.2 });
    const swings = bars.map((b) => Math.abs(b.high - b.low));
    expect([...swings].sort((a, b) => b - a)).toEqual(swings);
  });

  it('reports save rate and value per save as tied leaders, since net is linear in S·V', () => {
    for (const a of [{ valuePerSave: 120, costPerContact: 5, saveRate: 0.2 }, { valuePerSave: 200, costPerContact: 2, saveRate: 0.3 }]) {
      expect(sensitivity(1478, a).leaders.sort()).toEqual(['Save rate', 'Value per save']);
    }
  });

  it('names cost per contact alone when it dominates', () => {
    expect(sensitivity(10000, { valuePerSave: 20, costPerContact: 40, saveRate: 0.05 }).leaders).toEqual(['Cost per contact']);
  });
});

describe('tieredNet', () => {
  it('prices tier 1 with the user assumptions so it matches the single-tier result', () => {
    expect(tieredNet(150, 8, 0.3).net1).toBeCloseTo(net(1478, 150, 8, 0.3), 9);
  });
});

describe('logSpace', () => {
  it('spans the range inclusively', () => {
    const xs = logSpace(100, 50000, 25);
    expect(xs).toHaveLength(25);
    expect(xs[0]).toBe(100);
    expect(xs[24]).toBe(50000);
  });
});
