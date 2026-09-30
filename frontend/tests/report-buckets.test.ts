/**
 * Three-hour bucket mapping contract (reference reports layout).
 *
 * The reference daily chart shows eight 3-hour slots; the local API returns
 * 24 hourly values. This gate pins the aggregation so a change to either side
 * cannot silently reshape the chart.
 *
 * Run: npx tsx tests/report-buckets.test.ts
 */
import assert from "node:assert/strict";
import { toThreeHourBuckets } from "../src/report-buckets";

const labels = [
  "0:00~3:00",
  "3:00~6:00",
  "6:00~9:00",
  "9:00~12:00",
  "12:00~15:00",
  "15:00~18:00",
  "18:00~21:00",
  "21:00~24:00",
];

// ── 1. Always eight slots, in reference order ──────────────────────────────
{
  const buckets = toThreeHourBuckets({});
  assert.equal(buckets.length, 8, "must always render eight slots");
  assert.deepEqual(
    buckets.map((b) => b.hour),
    labels,
    "slot labels must match the reference order",
  );
  assert.ok(
    buckets.every((b) => b.minutes === 0),
    "an empty distribution renders eight zero slots, not an empty chart",
  );
}

// ── 2. Hours fold into the right bucket ───────────────────────────────────
{
  const buckets = toThreeHourBuckets({
    "0": 10, "1": 20, "2": 30, // → 0:00~3:00 = 60
    "3": 5, // → 3:00~6:00
    "11": 40, // → 9:00~12:00
    "23": 99, // → 21:00~24:00
  });
  assert.equal(buckets[0].minutes, 60, "hours 0-2 sum into the first slot");
  assert.equal(buckets[1].minutes, 5, "hour 3 starts the second slot");
  assert.equal(buckets[3].minutes, 40, "hour 11 lands in 9:00~12:00");
  assert.equal(buckets[7].minutes, 99, "hour 23 closes 21:00~24:00");
  assert.equal(buckets[2].minutes, 0, "unset hours stay zero");
}

// ── 3. Out-of-range and junk keys are ignored ─────────────────────────────
{
  const buckets = toThreeHourBuckets({ "24": 500, "-1": 500, junk: 500, "5": 7 });
  assert.equal(buckets.reduce((sum, b) => sum + b.minutes, 0), 7, "only in-range hours count");
}

// ── 4. Fractional minutes round rather than truncate ──────────────────────
{
  const buckets = toThreeHourBuckets({ "7": 0.6, "8": 0.6 });
  assert.equal(buckets[2].minutes, 1, "1.2 minutes rounds to 1");
}

console.log("report-buckets: all assertions passed");
