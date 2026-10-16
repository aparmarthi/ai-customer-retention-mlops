// Acklam's rational approximation of the inverse standard normal CDF.
export function inverseNormal(p: number): number {
  if (p <= 0 || p >= 1) return NaN;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pLow = 0.02425;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= 1 - pLow) {
    const q = p - 0.5;
    const r = q * q;
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

export interface SampleSizeResult {
  nPerArm: number;
  p1: number;
  p2: number;
  zAlpha: number;
  zBeta: number;
}

// Two-sided two-proportion z-test; treatment is expected to reduce churn by `mde` (absolute).
export function sampleSizePerArm(baseline: number, mde: number, alpha: number, power: number): SampleSizeResult | null {
  const p1 = baseline;
  const p2 = baseline - mde;
  if (!(p1 > 0 && p1 < 1 && p2 > 0 && p2 < 1 && mde > 0 && alpha > 0 && alpha < 1 && power > 0 && power < 1)) return null;
  const zAlpha = inverseNormal(1 - alpha / 2);
  const zBeta = inverseNormal(power);
  const pBar = (p1 + p2) / 2;
  const num = zAlpha * Math.sqrt(2 * pBar * (1 - pBar)) + zBeta * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2));
  return { nPerArm: Math.ceil((num * num) / (mde * mde)), p1, p2, zAlpha, zBeta };
}
