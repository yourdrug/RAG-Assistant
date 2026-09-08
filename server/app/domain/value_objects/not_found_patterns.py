"""Canonical patterns indicating the LLM answered "information not found".

Must stay in sync with the system prompt's "not found" phrasing.
If you change the canonical phrase in the prompt, update this tuple too.
"""

NOT_FOUND_PATTERNS: tuple[str, ...] = (
    "не найден",
    "не найдена",
    "не найдено",
    "нет информации",
    "не удалось найти",
    "информация не найдена",
    "в предоставленных документах",
    "в документах нет",
    "не обнаружен",
)
