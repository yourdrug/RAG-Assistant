"use client";
import { FileText, Search as SearchIcon } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { type SearchMode, useExactSearch } from "@/shared/api/hooks/use-search";
import { getSearchPagination } from "@/shared/lib/search-pagination";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";

const SEARCH_PAGE_SIZE = 50;

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
  const [search, setSearch] = useState({ query: "", mode: "exact" as SearchMode, offset: 0 });
  const isCurrentSearch = search.query === query.trim() && search.mode === mode;
  const hasSearched = isCurrentSearch && search.query.length >= 3;
  const exactSearch = useExactSearch({ ...search, limit: SEARCH_PAGE_SIZE }, hasSearched);
  const results = hasSearched ? (exactSearch.data?.results ?? []) : [];
  const total = hasSearched ? (exactSearch.data?.total ?? 0) : 0;
  const offset = exactSearch.data?.offset ?? search.offset;
  const pagination = getSearchPagination(total, offset, SEARCH_PAGE_SIZE);

  useEffect(() => {
    const timer = setTimeout(() => setSearch({ query: query.trim(), mode, offset: 0 }), 400);
    return () => clearTimeout(timer);
  }, [query, mode]);

  useEffect(() => {
    if (hasSearched && exactSearch.data && search.offset > 0 && search.offset >= total) {
      setSearch((value) => ({ ...value, offset: pagination.lastOffset }));
    }
  }, [hasSearched, exactSearch.data, search.offset, total, pagination.lastOffset]);

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
            {query.trim().length > 0 && query.trim().length < 3 && (
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
                Found {total} result{total !== 1 ? "s" : ""} for &quot;{search.query}
                &quot; ({mode === "exact" ? "pg_trgm ranked" : "contains"})
              </p>
              <p className="text-xs text-muted-foreground">
                Showing {offset + 1}–{offset + results.length} of {total}
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
                    {highlightMatch(r.content, search.query)}
                  </p>
                </div>
              ))}
            </div>
          )}

          {hasSearched && (pagination.pages > 1 || pagination.hasPrevious) && (
            <nav
              aria-label="Search result pages"
              className="flex items-center justify-between mt-6"
            >
              <span className="text-sm text-muted-foreground">
                Page {pagination.page}
                {pagination.pages >= pagination.page && ` of ${pagination.pages}`}
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  disabled={!pagination.hasPrevious || exactSearch.isFetching}
                  onClick={() =>
                    setSearch((value) => ({
                      ...value,
                      offset: Math.max(0, offset - SEARCH_PAGE_SIZE),
                    }))
                  }
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  disabled={!pagination.hasNext || exactSearch.isFetching}
                  onClick={() =>
                    setSearch((value) => ({ ...value, offset: offset + SEARCH_PAGE_SIZE }))
                  }
                >
                  Next
                </Button>
              </div>
            </nav>
          )}

          {query.trim().length >= 3 && (!hasSearched || exactSearch.isFetching) && (
            <p role="status" className="text-sm text-muted-foreground py-6">
              Searching…
            </p>
          )}

          {hasSearched && exactSearch.isError && (
            <p role="alert" className="text-sm text-destructive py-6">
              Search failed. Please try again.
            </p>
          )}

          {/* Empty state after search */}
          {hasSearched &&
            results.length === 0 &&
            exactSearch.isSuccess &&
            !exactSearch.isFetching && (
              <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                <SearchIcon className="h-12 w-12 mb-4 opacity-50" />
                <p className="text-lg">No results found</p>
                <p className="text-sm mt-1">Try a different query or switch search mode</p>
              </div>
            )}

          {/* Initial state */}
          {query.trim().length < 3 && (
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
