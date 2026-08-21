import { AlertTriangle, CheckCircle2, Info } from "lucide-react";

export function DryRunSuggestion({ suggestion }: { suggestion: string | null }) {
  if (!suggestion) return null;

  const isGood = suggestion.includes("хорошее") || suggestion.includes("хороший");
  const isEncoding = suggestion.includes("кодировк");

  return (
    <div
      className={`flex items-start gap-2 px-3 py-2 rounded-md text-sm ${
        isGood
          ? "bg-green-50 text-green-700 dark:bg-green-950/50 dark:text-green-400"
          : isEncoding
            ? "bg-orange-50 text-orange-700 dark:bg-orange-950/50 dark:text-orange-400"
            : "bg-yellow-50 text-yellow-700 dark:bg-yellow-950/50 dark:text-yellow-400"
      }`}
    >
      {isGood ? (
        <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0" />
      ) : isEncoding ? (
        <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
      ) : (
        <Info className="h-4 w-4 mt-0.5 shrink-0" />
      )}
      <span>{suggestion}</span>
    </div>
  );
}
