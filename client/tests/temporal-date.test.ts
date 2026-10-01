import assert from "node:assert/strict";
import { test } from "node:test";
import {
  readTemporalDate,
  validIsoDate,
  writeTemporalDate,
} from "../src/shared/lib/temporal-date.ts";

test("calendar dates survive reload and stay isolated when navigating to another conversation", () => {
  const url = writeTemporalDate(new URLSearchParams({ id: "7" }), {
    date: "2024-02-29",
    source: "calendar",
  });
  assert.deepEqual(readTemporalDate(new URLSearchParams(url.toString())), {
    date: "2024-02-29",
    source: "calendar",
  });
  assert.deepEqual(readTemporalDate(new URLSearchParams({ id: "8" })), {
    date: null,
    source: "calendar",
  });
  assert.equal(validIsoDate("2023-02-29"), null);
  assert.equal(validIsoDate("2024-13-01"), null);
});

test("a question date does not become a calendar selection and can be cleared", () => {
  const url = writeTemporalDate(new URLSearchParams({ id: "7" }), {
    date: "2024-01-01",
    source: "question",
  });
  assert.equal(readTemporalDate(url).source, "question");
  const cleared = writeTemporalDate(url, { date: null, source: "calendar" });
  assert.equal(cleared.toString(), "id=7");
});
