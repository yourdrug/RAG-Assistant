import assert from "node:assert/strict";
import { test } from "node:test";
import { AxiosError, AxiosHeaders } from "axios";
import {
  describeHttpError,
  getErrorMessage,
  shouldRetryRequest,
} from "../src/shared/lib/errors.ts";

test("Reads the current API error envelope and older validation responses", () => {
  assert.equal(
    describeHttpError(409, { message: "A group by that name already exists." }),
    "A group by that name already exists.",
  );
  assert.equal(
    describeHttpError(422, {
      message: "Ошибка валидации",
      errors: { "reasons.0": "Invalid reason" },
    }),
    "reasons.0: Invalid reason",
  );
  assert.equal(
    describeHttpError(422, { detail: [{ loc: ["body", "question"], msg: "Field required" }] }),
    "question: Field required",
  );
});

test("Uses status, Retry-After and network context without exposing technical messages", () => {
  const error = new AxiosError("technical message");
  error.response = {
    data: { message: "private details" },
    status: 429,
    statusText: "",
    headers: { "retry-after": "15" },
    config: { headers: new AxiosHeaders() },
  };

  assert.equal(getErrorMessage(error), "Too many requests. Try again in 15 seconds.");
  assert.equal(shouldRetryRequest(0, error), false);
  assert.match(getErrorMessage(new AxiosError("technical message", "ERR_NETWORK")), /connection/);
  assert.equal(shouldRetryRequest(0, new AxiosError("network", "ERR_NETWORK")), true);
  assert.doesNotMatch(describeHttpError(500, { message: "private details" }), /private details/);
  assert.doesNotMatch(describeHttpError(502, "<html>Bad gateway</html>"), /html/);
});
