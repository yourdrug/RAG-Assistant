import type { Source } from "../api/types";

interface MarkdownNode {
  type: string;
  value?: string;
  url?: string;
  title?: string;
  children?: MarkdownNode[];
}

export function documentUrl(source: Pick<Source, "document_id">): string | null {
  const id = source.document_id;
  return id != null && Number.isSafeInteger(id) && id > 0 ? `/documents/${id}` : null;
}

// Convert text nodes, never code, existing links or Markdown destinations.
// Historical answers lack the mapping and must not invent one from array order.
export function remarkDocumentCitations(sources: Source[]) {
  const byId = new Map(
    sources
      .filter((source) => source.citation_id != null)
      .map((source) => [source.citation_id, source]),
  );
  return () => (tree: MarkdownNode) => {
    function visit(node: MarkdownNode) {
      if (["code", "inlineCode", "link", "image", "linkReference"].includes(node.type)) return;
      if (!node.children) return;
      node.children = node.children.flatMap((child) => {
        if (child.type !== "text" || !child.value) {
          visit(child);
          return [child];
        }
        const result: MarkdownNode[] = [];
        const value = child.value;
        let position = 0;
        for (const match of value.matchAll(/\[(\d+)\]/g)) {
          const source = byId.get(Number(match[1]));
          const url = source && documentUrl(source);
          if (!source || !url) continue;
          const start = match.index;
          if (start > position) result.push({ type: "text", value: value.slice(position, start) });
          result.push({
            type: "link",
            url,
            title: source.source,
            children: [{ type: "text", value: match[0] }],
          });
          position = start + match[0].length;
        }
        if (position < value.length) result.push({ type: "text", value: value.slice(position) });
        return result;
      });
    }
    visit(tree);
  };
}
