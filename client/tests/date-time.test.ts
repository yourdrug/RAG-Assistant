import assert from "node:assert/strict";
import { test } from "node:test";
import {
  dateInputDisplay,
  dateInputValue,
  formatDate,
  formatDateTime,
  formatTime,
} from "../src/shared/lib/date-time.ts";

test("dates and times use dots, leading zeros, full years and 24-hour time", () => {
  const date = new Date(2026, 0, 2, 14, 45, 3);
  assert.equal(formatDate(date), "02.01.2026");
  assert.equal(formatTime(date), "14.45.03");
  assert.equal(formatDateTime(date), "02.01.2026 14.45.03");
  assert.equal(formatTime(new Date(2026, 0, 2)), "00.00.00");
});

test("calendar dates do not shift across timezones, timestamps use local time", () => {
  const previous = process.env.TZ;
  try {
    process.env.TZ = "America/Los_Angeles";
    assert.equal(formatDate("2024-02-29"), "29.02.2024");
    assert.equal(formatDateTime("2026-01-02T01:05:03Z"), "01.01.2026 17.05.03");
  } finally {
    if (previous === undefined) delete process.env.TZ;
    else process.env.TZ = previous;
  }
});

test("missing and invalid timestamps have a consistent fallback", () => {
  for (const value of [null, undefined, "", "invalid"]) {
    assert.equal(formatDate(value), "—");
    assert.equal(formatTime(value), "—");
    assert.equal(formatDateTime(value), "—");
  }
});

test("inputs display dotted dates and send valid ISO dates including seconds", () => {
  assert.equal(dateInputDisplay("2024-02-29"), "29.02.2024");
  assert.equal(dateInputDisplay("2024-02-29T14:45"), "29.02.2024 14.45.00");
  assert.equal(dateInputDisplay("2024-02-29T14:45:03"), "29.02.2024 14.45.03");
  assert.equal(dateInputValue("29.02.2024"), "2024-02-29");
  assert.equal(dateInputValue("29.02.2024 14.45.03", true), "2024-02-29T14:45:03");
  assert.equal(dateInputValue(""), "");
  assert.equal(dateInputDisplay(""), "");
  for (const value of ["29.02.2023", "31.04.2024", "12/31/2024", "01.0", "01.01.2024 00.00.00"]) {
    assert.equal(dateInputValue(value), null);
  }
  for (const value of [
    "01.01.2024",
    "01.01.2024 24.00.00",
    "01.01.2024 12.60.00",
    "01.01.2024 12.00.60",
  ]) {
    assert.equal(dateInputValue(value, true), null);
  }
});
