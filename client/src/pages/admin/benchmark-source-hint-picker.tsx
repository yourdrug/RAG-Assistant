import { Filter, X } from "lucide-react";
import { useRef, useState } from "react";
import { useSourceFiles } from "@/shared/api/hooks/use-benchmark";
import { Badge } from "@/shared/ui/badge";
import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";

export function SourceHintPicker({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (val: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const { data: files } = useSourceFiles(query || undefined);
  const [highlightIdx, setHighlightIdx] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = files || [];

  const selectFile = (file: string) => {
    const parts = (value || "").split(";");
    parts[parts.length - 1] = file;
    onChange(
      parts
        .map((part) => part.trim())
        .filter(Boolean)
        .join("; "),
    );
    setQuery("");
    setOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && highlightIdx >= 0 && filtered[highlightIdx]) {
      e.preventDefault();
      selectFile(filtered[highlightIdx]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <Input
          ref={inputRef}
          value={value || ""}
          onChange={(e) => {
            const v = e.target.value || null;
            onChange(v);
            setQuery((v || "").split(";").at(-1)?.trim() || "");
            setOpen(true);
            setHighlightIdx(-1);
          }}
          onFocus={() => {
            setOpen(true);
            setQuery((value || "").split(";").at(-1)?.trim() || "");
          }}
          onBlur={() => {
            // Delay to allow click on dropdown item
            setTimeout(() => setOpen(false), 200);
          }}
          onKeyDown={handleKeyDown}
          placeholder="file1.pdf; file2.pdf"
          className="min-w-0 flex-1"
          aria-label="Expected source documents"
          title={value || undefined}
        />
        {value && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-9 w-9 p-0 shrink-0"
            aria-label="Clear expected sources"
            title="Clear expected sources"
            onClick={() => {
              onChange(null);
              setQuery("");
              inputRef.current?.focus();
            }}
          >
            <X className="h-3 w-3" />
          </Button>
        )}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-9 w-9 p-0 shrink-0"
          aria-label="Browse source documents"
          title="Browse source documents"
          onClick={() => {
            setOpen(true);
            setQuery("");
            inputRef.current?.focus();
          }}
        >
          <Filter className="h-3 w-3" />
        </Button>
      </div>
      {open && filtered.length > 0 && (
        <div
          ref={listRef}
          className="absolute z-50 top-full mt-1 w-full bg-popover border rounded-md shadow-md max-h-[200px] overflow-auto"
        >
          {filtered.map((file, idx) => (
            <div
              key={file}
              className={`px-3 py-1.5 text-sm cursor-pointer truncate ${
                idx === highlightIdx ? "bg-accent text-accent-foreground" : "hover:bg-muted"
              }`}
              onMouseDown={(e) => {
                e.preventDefault();
                selectFile(file);
              }}
              onMouseEnter={() => setHighlightIdx(idx)}
            >
              {file}
            </div>
          ))}
        </div>
      )}
      {value && (
        <div className="mt-3 flex flex-wrap gap-2">
          {value
            .split(";")
            .map((source) => source.trim())
            .filter(Boolean)
            .map((source, index) => (
              <Badge
                key={`${source}-${index}`}
                variant="secondary"
                className="max-w-full whitespace-normal break-words px-2.5 py-1 text-xs font-normal leading-relaxed [overflow-wrap:anywhere]"
              >
                {source}
              </Badge>
            ))}
        </div>
      )}
      <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
        Separate sources with ;. A match with any source counts as a hit. Leave blank to skip Hit
        Rate and MRR.
      </p>
    </div>
  );
}
