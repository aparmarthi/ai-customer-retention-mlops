import { describe, expect, it } from 'vitest';
import { BOUNDS, DEFAULT_SCENARIO, decodeScenario, encodeScenario, hashQuery } from '@/lib/scenario';

describe('scenario links', () => {
  it('round-trips a scenario', () => {
    const s = { k: 3200, valuePerSave: 95.5, costPerContact: 2.25, saveRate: 0.15, subscribers: 2_500_000 };
    expect(decodeScenario(encodeScenario(s))).toEqual(s);
  });

  it('falls back to defaults for missing or garbage params', () => {
    expect(decodeScenario('')).toEqual(DEFAULT_SCENARIO);
    expect(decodeScenario('k=abc&v=&c=NaN')).toEqual(DEFAULT_SCENARIO);
  });

  it('clamps out-of-range values', () => {
    const s = decodeScenario('k=9999999&v=-4&s=5&subs=1');
    expect(s.k).toBe(BOUNDS.k.max);
    expect(s.valuePerSave).toBe(BOUNDS.valuePerSave.min);
    expect(s.saveRate).toBe(BOUNDS.saveRate.max);
    expect(s.subscribers).toBe(BOUNDS.subscribers.min);
  });

  it('extracts the query from a hash route', () => {
    expect(hashQuery('#/simulator?k=10')).toBe('k=10');
    expect(hashQuery('#/simulator')).toBe('');
  });
});
