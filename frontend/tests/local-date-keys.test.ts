/**
 * Local date-key helpers used by the "what happened today?" lookups (audit F1).
 *
 * Run with: npx tsx tests/local-date-keys.test.ts
 * No test framework: plain node asserts, deterministic, zero live clocks —
 * every case pins an explicit date so the suite cannot drift with the day.
 */
import assert from "node:assert/strict";
import {
  findDayByDate,
  localDateStr,
  mondayOf,
  parseLocalDate,
  previousLocalDateStr,
  shiftLocalDateStr,
} from "../src/date-utils.ts";

// ── 1. previousLocalDateStr across month / year / leap boundaries ──────────
assert.equal(previousLocalDateStr("2026-10-01"), "2026-09-30", "cross-month");
assert.equal(previousLocalDateStr("2026-09-30"), "2026-09-29", "plain day");
assert.equal(previousLocalDateStr("2026-01-01"), "2025-12-31", "cross-year");
assert.equal(previousLocalDateStr("2026-03-01"), "2026-02-28", "non-leap February");
assert.equal(previousLocalDateStr("2024-03-01"), "2024-02-29", "leap February");
assert.equal(previousLocalDateStr("2024-01-01"), "2023-12-31", "leap year + cross-year");
assert.equal(previousLocalDateStr("2026-09-01"), "2026-08-31", "cross-month the other way");

// ── 2. shiftLocalDateStr is a signed calendar shift, not ±24h ─────────────
assert.equal(shiftLocalDateStr("2026-09-30", -1), "2026-09-29");
assert.equal(shiftLocalDateStr("2026-09-30", 1), "2026-10-01");
assert.equal(shiftLocalDateStr("2026-09-30", -30), "2026-08-31", "a month back");
assert.equal(shiftLocalDateStr("2026-09-30", 365), "2027-09-30", "a year forward");
// Invalid input must not silently become today.
assert.equal(shiftLocalDateStr("not-a-date", -1), "not-a-date");

// ── 3. localDateStr uses local fields, not the UTC date ────────────────────
// Between 00:00 and 07:59 local (UTC+8) `toISOString()` yields the previous
// day; the helper must not.
const earlyLocal = new Date(2026, 8, 30, 0, 30, 0);
assert.equal(localDateStr(earlyLocal), "2026-09-30", "00:30 local is still the 30th");
const lateLocal = new Date(2026, 8, 29, 23, 30, 0);
assert.equal(localDateStr(lateLocal), "2026-09-29", "23:30 local is still the 29th");
const midnight = new Date(2026, 0, 1, 0, 0, 0);
assert.equal(localDateStr(midnight), "2026-01-01", "local midnight, new year");

// The UTC date at that instant differs from the local date whenever the
// machine sits ahead of UTC — which is the shipped configuration (UTC+8) and
// exactly the 00:00–07:59 window called out in date-utils.ts. On a machine at
// or behind UTC the two coincide, so the inequality is asserted only then;
// the local-field assertion above holds unconditionally in every timezone.
const utcDateAtThatInstant = earlyLocal.toISOString().slice(0, 10);
if (-earlyLocal.getTimezoneOffset() > 0) {
  assert.notEqual(localDateStr(earlyLocal), utcDateAtThatInstant);
}

// ── 4. parseLocalDate round-trips and rejects junk ─────────────────────────
assert.equal(parseLocalDate("2026-09-30") instanceof Date, true);
assert.equal(parseLocalDate("2026-00-00"), null, "zero month/day → null");
assert.equal(parseLocalDate("nonsense"), null, "unparseable → null");
assert.equal(parseLocalDate(""), null);
for (const value of ["2026-02-31", "2026-13-01", "2026-01-32", "2026-09-30-extra"]) {
  assert.equal(parseLocalDate(value), null, `invalid date must not roll over: ${value}`);
}
assert.equal(localDateStr(parseLocalDate("2026-02-28") as Date), "2026-02-28");
// Local midnight on a boundary resolves to that boundary, in any timezone.
assert.equal(localDateStr(new Date(2026, 11, 31, 0, 0, 0)), "2026-12-31");
assert.equal(previousLocalDateStr(localDateStr(new Date(2026, 11, 31, 0, 0, 0))), "2026-12-30");

// ── 5. findDayByDate: key lookup on a sparse series ────────────────────────
const sparse = [
  { date: "2026-09-24", value: 1 },
  { date: "2026-09-29", value: 77 },
];
assert.equal(findDayByDate(sparse, "2026-09-29")?.value, 77, "found by key");
assert.equal(findDayByDate(sparse, "2026-09-30"), undefined, "today absent → undefined, not the last row");
assert.equal(findDayByDate([], "2026-09-30"), undefined, "empty series");
assert.equal(findDayByDate(sparse, ""), undefined, "empty key");
assert.equal(
  findDayByDate(sparse, "2026-09-30") === sparse[sparse.length - 1],
  false,
  "absent today must never resolve to the tail entry",
);

// Unsorted input is still resolved correctly (ordering is not a contract).
const unsorted = [{ date: "2026-09-30", value: 3 }, { date: "2026-09-29", value: 1 }];
assert.equal(findDayByDate(unsorted, "2026-09-30")?.value, 3);
assert.equal(findDayByDate(unsorted, "2026-09-29")?.value, 1);

// ── 6. mondayOf stays consistent with the local calendar ───────────────────
assert.equal(mondayOf(new Date(2026, 8, 30)), "2026-09-28", "Wednesday → that week's Monday");
assert.equal(mondayOf(new Date(2026, 8, 28)), "2026-09-28", "Monday itself");

console.log("local-date-keys: all assertions passed");
