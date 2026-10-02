import assert from "node:assert/strict";
import { test } from "node:test";
import { documentUrl, remarkDocumentCitations } from "../src/shared/lib/document-citations.ts";

test("citations resolve by explicit document number, not filtered source array position", () => {
  const tree = {
    type: "root",
    children: [{ type: "paragraph", children: [{ type: "text", value: "Согласно [2] и [999]." }] }],
  };
  remarkDocumentCitations([{ citation_id: 2, document_id: 42, source: "rules.pdf", pages: [] }])()(
    tree,
  );
  const nodes = tree.children[0].children as { type: string; url?: string; value?: string }[];
  assert.equal(nodes.find((node) => node.type === "link")?.url, "/documents/42");
  assert.ok(nodes.some((node) => node.value?.includes("[999]")));
});

test("legacy sources do not invent a citation mapping and code and links stay intact", () => {
  const code = { type: "inlineCode", value: "array[2]" };
  const link = {
    type: "link",
    url: "https://example.org",
    children: [{ type: "text", value: "[2]" }],
  };
  const tree = { type: "root", children: [code, link, { type: "text", value: "[1]" }] };
  const original = structuredClone(tree);
  remarkDocumentCitations([{ document_id: 42, source: "legacy.pdf", pages: [] }])()(tree);
  assert.deepEqual(tree, original);
});

test("document URLs reject invalid identities", () => {
  for (const id of [-1, 0, 1.5, NaN, Infinity])
    assert.equal(documentUrl({ document_id: id }), null);
  assert.equal(documentUrl({}), null);
  assert.equal(documentUrl({ document_id: 7 }), "/documents/7");
});
