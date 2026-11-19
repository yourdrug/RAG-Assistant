import type { LLMReplayMode } from "@/shared/api/types";
import { Label } from "@/shared/ui/label";

export interface LLMReplayStats {
  mode: LLMReplayMode;
  hits: number;
  misses: number;
  recorded: number;
  store_errors: number;
  saved_input_tokens: number;
  saved_output_tokens: number;
}

const MODES: { value: LLMReplayMode; label: string; hint: string }[] = [
  {
    value: "off",
    label: "Выключен",
    hint: "Каждый вызов LLM (генерация, вспомогательные шаги, судья) идёт к провайдеру и оплачивается. Ничего не сохраняется.",
  },
  {
    value: "record",
    label: "Переиспользовать и записывать",
    hint: "Если точно такой же запрос (модель, параметры, промпт) уже был, ответ берётся из записи бесплатно. Новые запросы оплачиваются и сохраняются. Меняете порог реранкера — платите только за вызовы, чей вход изменился.",
  },
  {
    value: "replay_only",
    label: "Только записанные",
    hint: "Используются только сохранённые ответы; запрос без записи завершается ошибкой вместо оплаты. Прогон гарантированно бесплатный, но непокрытые вопросы получат ошибки оценки.",
  },
  {
    value: "refresh",
    label: "Обновить записи",
    hint: "Все вызовы идут к провайдеру и перезаписывают сохранённые ответы. Нужен после смены версии модели у провайдера под тем же именем.",
  },
];

export function BenchmarkLLMReplay({
  value,
  onChange,
  disabled,
}: {
  value: LLMReplayMode;
  onChange: (mode: LLMReplayMode) => void;
  disabled: boolean;
}) {
  const selected = MODES.find((mode) => mode.value === value) ?? MODES[0];

  return (
    <div className="space-y-2">
      <Label htmlFor="benchmark-llm-replay" className="text-xs">
        Повтор ответов LLM
      </Label>
      <select
        id="benchmark-llm-replay"
        value={value}
        onChange={(event) => onChange(event.target.value as LLMReplayMode)}
        disabled={disabled}
        className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50"
      >
        {MODES.map((mode) => (
          <option key={mode.value} value={mode.value}>
            {mode.label}
          </option>
        ))}
      </select>
      <p className="text-xs text-muted-foreground">{selected.hint}</p>
      {value !== "off" && (
        <p className="text-xs text-muted-foreground">
          Повторённый ответ — тот же сэмпл, что и при записи: для замера шума судьи и латентности
          выключайте режим. Эмбеддинги и реранкер не кэшируются. Записи с текстом документов
          хранятся в БД и удаляются, если не использовались (по умолчанию 90 дней).
        </p>
      )}
    </div>
  );
}

export function LLMReplaySummary({ stats }: { stats: LLMReplayStats | null | undefined }) {
  if (!stats || stats.mode === "off") return null;
  const mode = MODES.find((item) => item.value === stats.mode)?.label ?? stats.mode;

  return (
    <p className="text-xs text-muted-foreground mb-2">
      Повтор ответов LLM ({mode}): из записи {stats.hits}, без записи {stats.misses}, записано{" "}
      {stats.recorded}
      {stats.store_errors > 0 ? `, ошибок хранилища ${stats.store_errors}` : ""}. Сэкономлено{" "}
      {stats.saved_input_tokens.toLocaleString()} input /{" "}
      {stats.saved_output_tokens.toLocaleString()} output токенов. Токены выше включают повторённые
      вызовы.
    </p>
  );
}
