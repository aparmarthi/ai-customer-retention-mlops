export const toYmdInt = (iso: string): number | null => {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  return m ? Number(`${m[1]}${m[2]}${m[3]}`) : null;
};

const DAY_MS = 86_400_000;

export const ymdToDays = (v: number): number => {
  const s = String(Math.trunc(v));
  return Date.UTC(Number(s.slice(0, 4)), Number(s.slice(4, 6)) - 1, Number(s.slice(6, 8))) / DAY_MS;
};

export const addDaysYmd = (ymd: number, days: number): number => {
  const d = new Date((ymdToDays(ymd) + days) * DAY_MS);
  return d.getUTCFullYear() * 10000 + (d.getUTCMonth() + 1) * 100 + d.getUTCDate();
};

export const fromYmdInt = (v: unknown): string => {
  const s = String(v ?? '');
  return /^\d{8}$/.test(s) ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : '';
};
