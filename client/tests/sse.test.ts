import assert from "node:assert/strict";
import { test } from "node:test";
import { streamChat } from "../src/shared/lib/sse.ts";

test("SSE preserves event names and Unicode across network packet boundaries", async () => {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(
    'data: {"text":"Привет"}\n\nevent: done\r\ndata: {"conversation_id":7,"message_id":8,"sources":[]}\r\n\r\n',
  );
  const originalFetch = globalThis.fetch;
  const chunks: string[] = [];
  const done: number[] = [];
  const errors: string[] = [];
  globalThis.fetch = async (_url, options) => {
    assert.equal(new Headers(options?.headers).get("Authorization"), "Bearer token");
    const body = JSON.parse(options?.body as string);
    assert.equal(body.as_of_date, "2026-01-01");
    return new Response(
      new ReadableStream({
        start(controller) {
          for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
          controller.close();
        },
      }),
    );
  };
  try {
    await streamChat({
      question: "Q",
      token: "token",
      asOfDate: "2026-01-01",
      onChunk: (text) => chunks.push(text),
      onDone: (data) => done.push(data.conversation_id),
      onError: (error) => errors.push(error),
    });
    assert.deepEqual(chunks, ["Привет"]);
    assert.deepEqual(done, [7]);
    assert.deepEqual(errors, []);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("SSE reports HTTP failures without generating an answer", async () => {
  const originalFetch = globalThis.fetch;
  const errors: string[] = [];
  globalThis.fetch = async () =>
    new Response("", { status: 503, headers: { "Retry-After": "30" } });
  try {
    await streamChat({
      question: "Q",
      token: "token",
      onChunk: () => assert.fail("unexpected text"),
      onDone: () => assert.fail("unexpected done"),
      onError: (error) => errors.push(error),
    });
    assert.deepEqual(errors, ["The service is temporarily unavailable. Try again in 30 seconds."]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("SSE returns the resolved question date without sending an explicit calendar date", async () => {
  const originalFetch = globalThis.fetch;
  let resolvedDate: string | null | undefined;
  globalThis.fetch = async (_url, options) => {
    assert.equal(JSON.parse(options?.body as string).as_of_date, null);
    return new Response(
      'event: done\ndata: {"conversation_id":7,"message_id":null,"sources":null,"as_of_date":"2024-01-01"}\n\n',
    );
  };
  try {
    await streamChat({
      question: "Правила на 01.01.2024?",
      token: "token",
      asOfDate: null,
      onChunk: () => {},
      onDone: (data) => {
        resolvedDate = data.as_of_date;
      },
      onError: (error) => assert.fail(error),
    });
    assert.equal(resolvedDate, "2024-01-01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("SSE reports a lost connection and a stream closed before the final event", async () => {
  const originalFetch = globalThis.fetch;
  const errors: string[] = [];
  const options = {
    question: "Q",
    token: "token",
    onChunk: () => {},
    onDone: () => assert.fail("unexpected done"),
    onError: (error: string) => errors.push(error),
  };
  try {
    globalThis.fetch = async () => {
      throw new TypeError("Failed to fetch");
    };
    await streamChat(options);
    globalThis.fetch = async () => new Response('data: {"text":"partial"}\n\n');
    await streamChat(options);
    assert.deepEqual(errors, [
      "Could not reach the server. Check your connection and try again.",
      "The answer was interrupted. Please try again.",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("SSE delivers one application error and ignores an intentionally cancelled request", async () => {
  const originalFetch = globalThis.fetch;
  const errors: string[] = [];
  const options = {
    question: "Q",
    token: "token",
    onChunk: () => assert.fail("unexpected text"),
    onDone: () => assert.fail("unexpected done"),
    onError: (error: string) => errors.push(error),
  };
  try {
    globalThis.fetch = async () =>
      new Response(
        'event: error\ndata: {"code":"validation_error","error":"Choose a knowledge space."}\n\n',
      );
    await streamChat(options);
    const controller = new AbortController();
    controller.abort();
    globalThis.fetch = async () => {
      throw new DOMException("Cancelled", "AbortError");
    };
    await streamChat({ ...options, signal: controller.signal });
    assert.deepEqual(errors, ["Choose a knowledge space."]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("SSE preserves session expiry and the server's retry delay", async () => {
  const originalFetch = globalThis.fetch;
  const errors: Array<[string, number | undefined]> = [];
  const options = {
    question: "Q",
    token: "token",
    onChunk: () => assert.fail("unexpected text"),
    onDone: () => assert.fail("unexpected done"),
    onError: (error: string, status?: number) => errors.push([error, status]),
  };

  try {
    for (const payload of [
      { code: "unauthorized", error: "Unauthorized" },
      { code: "rate_limit_exceeded", retry_after: 12 },
      { code: "llm_unavailable", retry_after: 30 },
    ]) {
      globalThis.fetch = async () =>
        new Response(`event: error\ndata: ${JSON.stringify(payload)}\n\n`);
      await streamChat(options);
    }
    assert.deepEqual(errors, [
      ["Your session has expired. Please sign in again.", 401],
      ["Too many requests. Try again in 12 seconds.", undefined],
      ["The service is temporarily unavailable. Try again in 30 seconds.", undefined],
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
