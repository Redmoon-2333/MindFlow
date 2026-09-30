/**
 * Daily report chart aggregation: 24 hourly values → eight 3-hour slots.
 *
 * Pure on purpose: the reference daily chart draws eight fixed slots
 * (`0:00~3:00 … 21:00~24:00`) while the API returns per-hour minutes, so the
 * mapping lives here (no React, no fetch, no Date) and is covered by
 * `tests/report-buckets.test.ts`.
 */

export interface ThreeHourBucket {
  readonly hour: string;
  readonly minutes: number;
}

const SLOT_LABELS = [
  "0:00~3:00",
  "3:00~6:00",
  "6:00~9:00",
  "9:00~12:00",
  "12:00~15:00",
  "15:00~18:00",
  "18:00~21:00",
  "21:00~24:00",
] as const;

/** Bucket 24 hourly values into the reference's eight three-hour slots. */
export function toThreeHourBuckets(
  distribution: Readonly<Record<string, number>>,
): ThreeHourBucket[] {
  return SLOT_LABELS.map((hour, index) => {
    let minutes = 0;
    for (let h = index * 3; h < index * 3 + 3; h += 1) {
      const value = Number(distribution[String(h)]);
      if (Number.isFinite(value)) minutes += value;
    }
    return { hour, minutes: Math.round(minutes) };
  });
}
