export interface TemporalDate {
  date: string | null;
  source: "calendar" | "question";
}

export function validIsoDate(value: string | null): string | null {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value
    ? null
    : value;
}

export function readTemporalDate(params: URLSearchParams): TemporalDate {
  return {
    date: validIsoDate(params.get("as_of_date")),
    source: params.get("as_of_source") === "question" ? "question" : "calendar",
  };
}

export function writeTemporalDate(
  params: URLSearchParams,
  selection: TemporalDate,
): URLSearchParams {
  const next = new URLSearchParams(params);
  next.delete("as_of_date");
  next.delete("as_of_source");
  const date = validIsoDate(selection.date);
  if (date) {
    next.set("as_of_date", date);
    if (selection.source === "question") next.set("as_of_source", "question");
  }
  return next;
}
