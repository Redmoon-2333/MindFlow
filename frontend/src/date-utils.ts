/**
 * Local-timezone date helpers.
 *
 * The backend computes "today" in the configured local timezone (e.g.
 * business_today()), so the frontend must send local YYYY-MM-DD strings.
 * Using Date.prototype.toISOString() here returns the UTC date, which is
 * wrong for UTC+8 users between 00:00 and 07:59 local time (it would be
 * the previous day). These helpers always derive the date from the local
 * clock fields instead.
 */

/** Local YYYY-MM-DD for the given Date (defaults to now). */
export function localDateStr(d: Date = new Date()): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** Local YYYY-MM-DD of the Monday of the week containing *date*. */
export function mondayOf(date: Date = new Date()): string {
  const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const dow = (d.getDay() + 6) % 7; // 0=Monday .. 6=Sunday
  d.setDate(d.getDate() - dow);
  return localDateStr(d);
}

/** Weekday label (0=Sunday..6=Saturday) of a YYYY-MM-DD string. */
export function dayLabel(dateStr: string, short = false): string {
  const names = short
    ? ["日", "一", "二", "三", "四", "五", "六"]
    : ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
  const [y, m, d] = dateStr.split("-").map(Number);
  if (!y || !m || !d) return "";
  const date = new Date(y, m - 1, d);
  return names[date.getDay()];
}

/** Parse a YYYY-MM-DD string into a local Date at midnight. */
export function parseLocalDate(dateStr: string): Date | null {
  const [y, m, d] = dateStr.split("-").map(Number);
  if (!y || !m || !d) return null;
  const date = new Date(y, m - 1, d);
  return Number.isNaN(date.getTime()) ? null : date;
}