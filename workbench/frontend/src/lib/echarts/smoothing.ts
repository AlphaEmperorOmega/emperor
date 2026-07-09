/**
 * TensorBoard-style exponential moving average with debiasing. Returns a copy of
 * `points` with each `value` replaced by its smoothed value; all other fields
 * (step, wallTime, ...) are preserved. Non-finite values pass through untouched.
 * A `weight` of 0 is a no-op; values approach a flat line as it nears 1.
 */
export function applyEmaSmoothing<T extends { value: number }>(
  points: readonly T[],
  weight: number,
): T[] {
  const clamped = Math.min(Math.max(weight, 0), 1);
  if (clamped <= 0) {
    return points.map((point) => ({ ...point }));
  }
  let last = 0;
  let count = 0;
  return points.map((point) => {
    if (!Number.isFinite(point.value)) {
      return { ...point };
    }
    last = last * clamped + (1 - clamped) * point.value;
    count += 1;
    const debias = 1 - Math.pow(clamped, count);
    const smoothed = debias > 0 ? last / debias : last;
    return { ...point, value: smoothed };
  });
}
