"use client";
import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/shared/ui/button";
import type { DryRunResponse } from "@/shared/api/types";
import { DryRunSummaryBar } from "./DryRunSummaryBar";
import { DryRunSuggestion } from "./DryRunSuggestion";
import { DryRunCharsChart } from "./DryRunCharsChart";
import { DryRunPageList } from "./DryRunPageList";
import { DryRunPageDetail } from "./DryRunPageDetail";
import { DryRunActions } from "./DryRunActions";

export function DryRunResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [result, setResult] = useState<DryRunResponse | null>(
    () => (location.state as DryRunResponse | null) ?? null,
  );
  const [selectedPage, setSelectedPage] = useState<number | null>(null);

  const selectedPageData = useMemo(
    () => result?.pages.find((p) => p.page === selectedPage) ?? null,
    [result, selectedPage],
  );

  if (!result) {
    return (
      <div className="p-6 space-y-4">
        <Button variant="ghost" onClick={() => navigate(-1)}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <p className="text-muted-foreground">
          No dry-run results. Go to{" "}
          <a href="/admin/quality" className="underline">
            Quality
          </a>{" "}
          to run a preview.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <h1 className="text-xl font-bold">Dry-Run: {result.filename}</h1>
      </div>

      {/* Summary */}
      <DryRunSummaryBar result={result} />

      {/* Suggestion */}
      <DryRunSuggestion suggestion={result.suggestion} />

      {/* Chars chart */}
      <DryRunCharsChart
        pages={result.pages}
        selectedPage={selectedPage}
        onSelectPage={setSelectedPage}
      />

      {/* Split view: page list + detail */}
      <div className="flex gap-4 items-stretch">
        <div className="relative w-64 border rounded-md shrink-0">
          <div className="absolute inset-0 overflow-y-auto scrollbar-thin">
            <DryRunPageList
              pages={result.pages}
              selectedPage={selectedPage}
              onSelectPage={setSelectedPage}
            />
          </div>
        </div>

        <div className="flex-1 border rounded-md overflow-y-auto ">
          <DryRunPageDetail
            page={selectedPageData}
            previewId={result.preview_id}
          />
        </div>
      </div>

      {/* Actions */}
      <DryRunActions
        result={result}
        onResultUpdate={setResult}
        onReset={() => navigate("/admin/quality")}
      />
    </div>
  );
}
