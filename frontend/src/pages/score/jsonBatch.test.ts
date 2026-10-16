import { describe, expect, it } from 'vitest';
import { SAMPLES } from '@/data/samples';
import { EXAMPLE_BATCH_SIZE } from '@/data/profiles';
import { demoProbabilities, exampleBatchJson, parseJsonRecords } from '@/pages/score/jsonBatch';

describe('parseJsonRecords', () => {
  it('accepts a bare array and a {records} wrapper', () => {
    const a = parseJsonRecords('[{"a":1},{"b":"x"}]');
    const b = parseJsonRecords('{"records":[{"a":1}]}');
    expect(a.ok && a.records.length).toBe(2);
    expect(a.ok && a.headers).toEqual(['a', 'b']);
    expect(b.ok && b.records).toEqual([{ a: 1 }]);
  });

  it('drops nulls and converts booleans', () => {
    const r = parseJsonRecords('[{"a":null,"b":true,"c":false}]');
    expect(r.ok && r.records[0]).toEqual({ b: 1, c: 0 });
  });

  it('rejects malformed input with a readable message', () => {
    for (const bad of ['', 'not json', '{"x":1}', '[]', '[1,2]', '[{"a":{"nested":1}}]']) {
      const r = parseJsonRecords(bad);
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.error.length).toBeGreaterThan(0);
    }
  });
});

describe('demoProbabilities', () => {
  it('uses recorded scores for the bundled samples in any order', () => {
    const reversed = [...SAMPLES].reverse();
    const out = demoProbabilities(reversed.map((s) => s.record));
    expect(out.estimated).toBe(false);
    expect(out.probs).toEqual(reversed.map((s) => s.expected));
  });

  it('estimates unknown records and flags them', () => {
    const out = demoProbabilities([{ foo: 1 }]);
    expect(out.estimated).toBe(true);
    expect(out.probs[0]).toBeGreaterThan(0);
  });

  it('the default example parses cleanly', () => {
    const parsed = parseJsonRecords(exampleBatchJson());
    expect(parsed.ok && parsed.records.length).toBe(EXAMPLE_BATCH_SIZE);
    expect(parsed.ok && parsed.headers.slice(0, 2)).toEqual(['customer_id', 'segment']);
  });
});
