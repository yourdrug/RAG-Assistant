type DateValue = string | number | Date | null | undefined;

function parseDate(value: DateValue): Date | null {
  if (value === null || value === undefined || value === "") return null;
  // Calendar dates have no timezone and must not shift to the previous day.
  const date =
    typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value)
      ? new Date(`${value}T00:00:00`)
      : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function formatDate(value: DateValue): string {
  const date = parseDate(value);
  return date
    ? `${pad(date.getDate())}.${pad(date.getMonth() + 1)}.${String(date.getFullYear()).padStart(4, "0")}`
    : "—";
}

export function formatTime(value: DateValue): string {
  const date = parseDate(value);
  return date ? `${pad(date.getHours())}.${pad(date.getMinutes())}.${pad(date.getSeconds())}` : "—";
}

export function formatDateTime(value: DateValue): string {
  return parseDate(value) ? `${formatDate(value)} ${formatTime(value)}` : "—";
}

/** Format ISO input values without applying a timezone conversion. */
export function dateInputDisplay(value: string): string {
  if (!value) return "";
  const [date, time] = value.split("T");
  const [year, month, day] = date.split("-");
  return `${day}.${month}.${year}${time ? ` ${time.replaceAll(":", ".").padEnd(8, ".00")}` : ""}`;
}

/** Return null while an input is incomplete or invalid; API values stay ISO. */
export function dateInputValue(value: string, withTime = false): string | null {
  if (!value) return "";
  const match = /^(\d{2})\.(\d{2})\.(\d{4})(?: (\d{2})\.(\d{2})\.(\d{2}))?$/.exec(value);
  if (!match || Boolean(match[4]) !== withTime) return null;
  const [, day, month, year, hour, minute, second] = match;
  const date = `${year}-${month}-${day}`;
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== date) return null;
  if (!withTime) return date;
  if (Number(hour) > 23 || Number(minute) > 59 || Number(second) > 59) return null;
  return `${date}T${hour}:${minute}:${second}`;
}
