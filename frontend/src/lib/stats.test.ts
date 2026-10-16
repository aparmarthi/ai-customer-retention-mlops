import { describe, expect, it } from 'vitest';
import { inverseNormal, sampleSizePerArm } from '@/lib/stats';

describe('inverseNormal', () => {
  it('matches standard z-scores', () => {
    expect(inverseNormal(0.5)).toBeCloseTo(0, 6);
    expect(inverseNormal(0.975)).toBeCloseTo(1.959964, 5);
    expect(inverseNormal(0.8)).toBeCloseTo(0.841621, 5);
    expect(inverseNormal(0.01)).toBeCloseTo(-2.326348, 5);
  });

  it('rejects probabilities outside (0, 1)', () => {
    expect(inverseNormal(0)).toBeNaN();
    expect(inverseNormal(1)).toBeNaN();
  });
});

describe('sampleSizePerArm', () => {
  it('returns null for invalid inputs', () => {
    expect(sampleSizePerArm(0.18, 0, 0.05, 0.8)).toBeNull();
    expect(sampleSizePerArm(0.02, 0.03, 0.05, 0.8)).toBeNull();
    expect(sampleSizePerArm(0.18, 0.03, 0, 0.8)).toBeNull();
  });

  it('needs more users for smaller effects', () => {
    const big = sampleSizePerArm(0.18, 0.05, 0.05, 0.8)!.nPerArm;
    const small = sampleSizePerArm(0.18, 0.02, 0.05, 0.8)!.nPerArm;
    expect(small).toBeGreaterThan(big);
  });
});
