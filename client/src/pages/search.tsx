"use client";
import { FileText, Search as SearchIcon } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { type SearchMode, useExactSearch } from "@/shared/api/hooks/use-search";
import type { ExactSearchResult } from "@/shared/api/types";
import { Input } from "@/shared/ui/input";

function highlightMatch(text: string, query: string): ReactNode[] {
  if (!query.trim()) return [text];
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escaped})`, "gi");
  const parts = text.split(regex);
  return parts.map((part, i) =>
    regex.test(part) ? (
      <mark key={i} className="bg-yellow-200 dark:bg-yellow-800 rounded px-0.5">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("exact");
  const [results, setResults] = useState<ExactSearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const exactSearch = useExactSearch();
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const queryRef = useRef(query);
  const modeRef = useRef(mode);
  queryRef.current = query;
  modeRef.current = mode;

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (query.trim().length < 3) {
      setResults([]);
      setHasSearched(false);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setHasSearched(true);
      try {
        const res = await exactSearch.mutateAsync({
          query: queryRef.current,
          mode: modeRef.current,
          limit: 50,
        });
        setResults(res.results);
      } catch {
        setResults([]);
      }
    }, 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, mode]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-4">
          <div>
            <h1 className="text-2xl font-bold">Search</h1>
            <p className="text-sm text-muted-foreground">
              Search across all indexed document chunks
            </p>
          </div>

          {/* Search bar */}
          <div className="space-y-2">
            <div className="relative">
              <SearchIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            {query.length > 0 && query.length < 3 && (
              <p className="text-xs text-muted-foreground">Minimum 3 characters to search</p>
            )}
          </div>

          {/* Mode toggle */}
          <div className="flex items-center gap-4">
            <span className="text-xs text-muted-foreground">Mode:</span>
            <label className="flex items-center gap-1.5 cursor-pointer text-sm">
              <input
                type="radio"
                name="searchMode"
                value="exact"
                checked={mode === "exact"}
                onChange={() => setMode("exact")}
                className="w-3.5 h-3.5"
              />
              <span>Exact (pg_trgm ranked)</span>
            </label>
            <label className="flex items-center gap-1.5 cursor-pointer text-sm">
              <input
                type="radio"
                name="searchMode"
                value="icontains"
                checked={mode === "icontains"}
                onChange={() => setMode("icontains")}
                className="w-3.5 h-3.5"
              />
              <span>Contains (ILIKE)</span>
            </label>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto">
          {/* Results list */}
          {hasSearched && results.length > 0 && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                Found {results.length} result{results.length !== 1 ? "s" : ""} for &quot;{query}
                &quot; ({mode === "exact" ? "pg_trgm ranked" : "contains"})
              </p>
              {results.map((r) => (
                <div
                  key={r.chunk_id}
                  className="border rounded-lg p-4 space-y-2 hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium text-primary">{r.filename}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">chunk #{r.chunk_index}</span>
                  </div>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {highlightMatch(r.content, query)}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* Empty state after search */}
          {hasSearched && results.length === 0 && query.length >= 3 && !exactSearch.isPending && (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <SearchIcon className="h-12 w-12 mb-4 opacity-50" />
              <p className="text-lg">No results found</p>
              <p className="text-sm mt-1">Try a different query or switch search mode</p>
            </div>
          )}

          {/* Initial state */}
          {!hasSearched && (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <SearchIcon className="h-12 w-12 mb-4 opacity-50" />
              <p className="text-lg">Search your documents</p>
              <p className="text-sm mt-1">Type at least 3 characters to start searching</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
