"use client";
import { useState } from "react";
import { Input } from "@/shared/ui/input";
import { Button } from "@/shared/ui/button";
import { Search as SearchIcon } from "lucide-react";

export function SearchPage() {
  const [query, setQuery] = useState("");
  return (
    <div className="p-6 space-y-6">
      <div><h1 className="text-2xl font-bold">Search</h1><p className="text-muted-foreground">Search across documents and chats</p></div>
      <div className="flex gap-2 max-w-2xl">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input placeholder="Search documents, chats, content..." value={query} onChange={(e) => setQuery(e.target.value)} className="pl-9" />
        </div>
        <Button>Search</Button>
      </div>
      <div className="text-center py-12 text-muted-foreground">
        <SearchIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p className="text-lg">Full-text search coming soon</p>
        <p className="text-sm mt-1">Requires a backend search endpoint</p>
      </div>
    </div>
  );
}
