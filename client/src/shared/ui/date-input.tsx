import { CalendarDays } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { dateInputDisplay, dateInputValue } from "@/shared/lib/date-time";
import { cn } from "@/shared/lib/utils";
import { Input } from "@/shared/ui/input";

interface DateInputProps {
  value: string;
  onValueChange: (value: string) => void;
  withTime?: boolean;
  max?: string;
  disabled?: boolean;
  title?: string;
  className?: string;
}

export function DateInput({
  value,
  onValueChange,
  withTime = false,
  max,
  disabled,
  title,
  className,
}: DateInputProps) {
  const [draft, setDraft] = useState(() => dateInputDisplay(value));
  const picker = useRef<HTMLInputElement>(null);
  useEffect(() => setDraft(dateInputDisplay(value)), [value]);

  return (
    <div className="relative flex items-center">
      <Input
        value={draft}
        disabled={disabled}
        title={title}
        aria-label={title ?? (withTime ? "Дата и время" : "Дата")}
        placeholder={withTime ? "дд.мм.гггг чч.мм.сс" : "дд.мм.гггг"}
        className={cn(className, "pr-9")}
        onChange={(event) => {
          const text = event.target.value;
          setDraft(text);
          const next = dateInputValue(text, withTime);
          if (next !== null && (!max || next <= max)) onValueChange(next);
        }}
        onBlur={() => setDraft(dateInputDisplay(value))}
      />
      <input
        ref={picker}
        type={withTime ? "datetime-local" : "date"}
        step={withTime ? 1 : undefined}
        value={value}
        max={max}
        disabled={disabled}
        tabIndex={-1}
        aria-hidden="true"
        className="absolute right-0 bottom-0 h-0 w-0 opacity-0"
        onChange={(event) => onValueChange(event.target.value)}
      />
      <button
        type="button"
        disabled={disabled}
        aria-label="Выбрать дату в календаре"
        className="absolute right-2 text-muted-foreground disabled:opacity-50"
        onClick={() => picker.current?.showPicker()}
      >
        <CalendarDays className="h-4 w-4" />
      </button>
    </div>
  );
}
