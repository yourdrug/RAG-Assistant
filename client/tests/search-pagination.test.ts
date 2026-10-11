import assert from "node:assert/strict";
import { test } from "node:test";
import { getSearchPagination } from "../src/shared/lib/search-pagination.ts";

test("the last partial page can return to earlier matches", () => {
  assert.deepEqual(getSearchPagination(53, 50, 50), {
    page: 2,
    pages: 2,
    hasPrevious: true,
    hasNext: false,
    lastOffset: 50,
  });
});

test("a shrinking total keeps a route back and identifies the last valid page", () => {
  assert.deepEqual(getSearchPagination(50, 50, 50), {
    page: 2,
    pages: 1,
    hasPrevious: true,
    hasNext: false,
    lastOffset: 0,
  });
});

test("a failed later page with no total still permits returning to the first page", () => {
  assert.equal(getSearchPagination(0, 50, 50).hasPrevious, true);
  assert.equal(getSearchPagination(0, 0, 50).hasPrevious, false);
});

test("next navigation follows the full total rather than the current result count", () => {
  assert.equal(getSearchPagination(120, 0, 50).hasNext, true);
  assert.equal(getSearchPagination(120, 50, 50).hasNext, true);
  assert.equal(getSearchPagination(120, 100, 50).hasNext, false);
});
