const usd0 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
const usd2 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const usdCompact = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', maximumSignificantDigits: 3 });
const usdCompact2 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });

const withSign = (x: number, s: string) => (x < 0 ? `\u2212${s.replace('-', '')}` : s);

export const fmtUsd = (x: number): string => withSign(x, usd0.format(Math.round(x)));
export const fmtUsd2 = (x: number): string => withSign(x, usd2.format(x));
export const fmtUsdCompact = (x: number): string => withSign(x, usdCompact.format(x));
export const fmtUsdMillions = (x: number): string =>
  Math.abs(x) >= 1e6 ? withSign(x, usdCompact2.format(x)) : fmtUsdCompact(x);
export const fmtSignedUsdCompact = (x: number): string => (x > 0 ? `+${fmtUsdCompact(x)}` : fmtUsdCompact(x));
export const fmtInt = (x: number): string => withSign(x, int.format(Math.round(x)));
export const fmtCompact = (x: number): string => compact.format(x);
export const fmtPct = (x: number, digits = 1): string =>
  withSign(x, `${(x * 100).toFixed(digits)}%`);
export const fmtMultiple = (x: number, digits = 1): string => `${x.toFixed(digits)}\u00d7`;
export const fmtProb = (x: number, digits = 3): string => x.toFixed(digits);
export const fmtMs = (x: number): string => `${int.format(Math.round(x))} ms`;
