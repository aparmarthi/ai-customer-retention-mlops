import { describe, expect, it } from 'vitest';
import { EXAMPLE_BATCH, EXAMPLE_BATCH_SIZE, PROFILE_SAMPLES } from '@/data/profiles';
import { FEATURES } from '@/data/features';
import { SAMPLES } from '@/data/samples';
import { demoEstimate } from '@/lib/demoScore';
import { riskBand } from '@/pages/score/risk';

const bandCounts = (ps: number[]) =>
  ps.reduce<Record<string, number>>((acc, p) => ({ ...acc, [riskBand(p)]: (acc[riskBand(p)] ?? 0) + 1 }), {});

describe('demo profiles', () => {
  it('cover every feature the model expects', () => {
    for (const s of PROFILE_SAMPLES) expect(Object.keys(s.record).sort()).toEqual([...FEATURES].sort());
  });

  it('single-customer profiles span all three risk bands', () => {
    const counts = bandCounts(PROFILE_SAMPLES.map((s) => s.expected));
    expect(counts.monitor).toBeGreaterThanOrEqual(3);
    expect(counts.capacity).toBeGreaterThanOrEqual(2);
    expect(counts.contact).toBeGreaterThanOrEqual(2);
  });

  it('example batch is deterministic, uniquely identified and well spread', () => {
    expect(EXAMPLE_BATCH).toHaveLength(EXAMPLE_BATCH_SIZE);
    expect(new Set(EXAMPLE_BATCH.map((r) => r.customer_id)).size).toBe(EXAMPLE_BATCH_SIZE);
    const counts = bandCounts(EXAMPLE_BATCH.map(demoEstimate));
    expect(counts.monitor).toBeGreaterThanOrEqual(15);
    expect(counts.capacity).toBeGreaterThanOrEqual(10);
    expect(counts.contact).toBeGreaterThanOrEqual(10);
  });
});

describe('demoEstimate', () => {
  it('stays within (0, 1) and tolerates missing fields', () => {
    expect(demoEstimate({})).toBeGreaterThan(0);
    expect(demoEstimate({})).toBeLessThan(1);
    for (const s of SAMPLES) {
      const p = demoEstimate(s.record);
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
  });

  it('ranks an expired one-off buyer above a loyal auto-renewer', () => {
    const [loyal] = PROFILE_SAMPLES;
    const expired = PROFILE_SAMPLES[PROFILE_SAMPLES.length - 1];
    expect(demoEstimate(expired.record)).toBeGreaterThan(demoEstimate(loyal.record) + 0.5);
  });
});
