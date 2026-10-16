import { SAMPLES, type CustomerRecord } from '@/data/samples';
import { EXAMPLE_BATCH } from '@/data/profiles';
import { demoEstimate } from '@/lib/demoScore';

export const MAX_JSON_RECORDS = 50_000;

export type JsonParseResult =
  | { ok: true; records: CustomerRecord[]; headers: string[] }
  | { ok: false; error: string };

function toRecord(x: unknown, i: number): CustomerRecord | string {
  if (!x || typeof x !== 'object' || Array.isArray(x)) return `Item ${i + 1} is not an object.`;
  const rec: CustomerRecord = {};
  for (const [key, v] of Object.entries(x)) {
    if (v === null || v === undefined) continue;
    if (typeof v === 'number' || typeof v === 'string') rec[key] = v;
    else if (typeof v === 'boolean') rec[key] = v ? 1 : 0;
    else return `Item ${i + 1}, field "${key}" must be a number, string, boolean or null.`;
  }
  return rec;
}

export function parseJsonRecords(text: string): JsonParseResult {
  if (text.trim() === '') return { ok: false, error: 'Paste a JSON array of customer records.' };
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: `Invalid JSON: ${e instanceof Error ? e.message : 'could not parse'}` };
  }
  const list = Array.isArray(data)
    ? data
    : data && typeof data === 'object' && Array.isArray((data as { records?: unknown }).records)
      ? (data as { records: unknown[] }).records
      : null;
  if (!list) return { ok: false, error: 'Expected an array of records, or an object like {"records": [...]}.' };
  if (list.length === 0) return { ok: false, error: 'The array is empty. Add at least one customer.' };
  if (list.length > MAX_JSON_RECORDS) return { ok: false, error: `Too many records (max ${MAX_JSON_RECORDS.toLocaleString()}). Use CSV upload for larger batches.` };

  const records: CustomerRecord[] = [];
  const headers = new Set<string>();
  for (let i = 0; i < list.length; i++) {
    const r = toRecord(list[i], i);
    if (typeof r === 'string') return { ok: false, error: r };
    Object.keys(r).forEach((h) => headers.add(h));
    records.push(r);
  }
  return { ok: true, records, headers: [...headers] };
}

export const exampleBatchJson = (): string => JSON.stringify(EXAMPLE_BATCH, null, 2);

const recordedScore = (r: CustomerRecord): number | undefined =>
  SAMPLES.find((s) => {
    const keys = Object.keys(s.record);
    return keys.length === Object.keys(r).length && keys.every((k) => r[k] === s.record[k]);
  })?.expected;

// Demo Mode: recorded model scores for the bundled samples, rule-of-thumb estimates for everything else.
export function demoProbabilities(records: CustomerRecord[]): { probs: number[]; estimated: boolean } {
  let estimated = false;
  const probs = records.map((r) => {
    const recorded = recordedScore(r);
    if (recorded !== undefined) return recorded;
    estimated = true;
    return demoEstimate(r);
  });
  return { probs, estimated };
}
