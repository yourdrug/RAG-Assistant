# План: Разделение глобального конфига на подконфиги по подсистемам

## Контекст

`server/app/config.py` — единый `Settings` (~100 полей, 23 секции), синглтон
`settings = Settings()` создаётся на уровне модуля, импортируется напрямую
в ~40 файлах всех слоёв приложения.

Цель: изолировать конфиги подсистем так, чтобы каждая подсистема видела только
необходимые ей параметры (например, RAG-пайплайн не должен иметь доступа к
конфигам S3, JWT, DB-кредам и admin-паролям).

### Подтверждённые проблемы

| # | Проблема | Где |
|---|----------|-----|
| P1 | RAG-пайплайн видит S3/JWT/DB/admin-credentials — не нужны для его работы | `infrastructure/ml/rag/**` |
| P2 | Application-слой импортирует `config` и `infrastructure.*` напрямую (layering violation) | `ingestion_orchestrator.py`, `benchmark_orchestrator.py`, `retrieval.py` |
| P3 | Hot-reload мутирует общий синглтон через `setattr` без locking → несогласованные снапшоты и partial reads для list-полей | `config_subscribers.py::apply_to_settings` |
| P4 | Тест любого модуля, транзитивно импортирующего `config`, требует ВСЕ обязательные env-переменные | ~40 файлов |
| P5 | `ocr_min_confidence` читается через `getattr`, но не объявлен в Settings → молча игнорируется из .env | `infrastructure/ml/ingestion/ocr.py:93,110` |
| P6 | Несогласованность hot-reload: `s3_access_key`/`s3_secret_key` одновременно в SENSITIVE_KEYS и DYNAMIC_PARAMS — кэш хранилища инвалидируется, но креды не обновляются → клиент хранилища работает со старыми кредами | `config_subscribers.py` |
| P7 | Типы: `db_port: str` (vs `redis_port: int`), `ocr_lang_surya: list` без `[str]` | `config.py` |

---

## Принятые архитектурные решения (trade-offs)

### D1. Гранулярность — по подсистемам

| Вариант | Плюсы | Минусы | Вердикт |
|---|---|---|---|
| A. По слоям (Domain/Infra/ML/Auth) | Просто начать | Концептуально неверно: RAG-конфиг нужен и в domain (снапшот), и в infra (адаптер). Слой ≠ потребитель конфига | ❌ |
| **B. По подсистемам** (DB, S3, RAG, LLM, Auth, ...) | Прямо решает цель «RAG не видит S3»; совпадает с существующими портами (`RagSettingsPort`, `ChatSettingsPort`) | Больше классов | ✅ |
| C. Bounded contexts (DDD) | Строгие границы | Переусложнение для монолита с одной БД | ❌ |

### D2. Механизм — flat `Settings` как приватный загрузчик + frozen dataclass подконфиги как публичный API

| Вариант | Плюсы | Минусы | Вердикт |
|---|---|---|---|
| A. Nested pydantic (`env_nested_delimiter="__"`) | Идиоматично для pydantic | **Ломает формат `.env`** (`DB_HOST` → `DB__HOST`) → миграция docker-compose interpolation, Taskfile `dotenv:`, `.env`, `.env.secrets`, оба `.env.example`, `deploy.sh`. Высокий риск, нулевая функциональная выгода | ❌ |
| **B. Flat Settings (приватный) + builder-функции → frozen dataclasses** | Ноль изменений в env-формате; подконфиг видит только свои поля; тестируется inline; env-механика и валидация секретов (`_check_security`, stage=prod) остаются в одном месте | Два уровня представления (~90 полей в 2 местах) — митигируется тестом-реестром (Phase 0.3) | ✅ |
| C. Подконфиги сами читают env | Нет дублирования | Каждый подконфиг знает про env-механику; невозможно совместно валидировать секреты | ❌ |

Ключевой аргумент против A: ценность разделения — в том, **что видит потребитель**
(изоляция через DI), а не в том, **как хранится источник**. Цель «RAG не видит S3»
достигается инъекцией подконфига, а не форматом env-файла.

### D3. Hot-reload — atomic swap вместо `setattr` на общем объекте

Текущая схема (`config_subscribers.py`): приходит `ConfigParameterChanged` →
`setattr(settings, "rerank_min_score", 0.2)`. Все запросы читают один мутируемый
объект. Проблемы:

- запрос, прочитавший параметры в начале и в конце обработки, может получить
  **половину старой и половину новой конфигурации** — несогласованный снапшот;
- для list-полей (`ocr_lang_surya`) возможен partial read при реассигне;
- лок решает, но добавляет lock-contention на каждый запрос.

Целевая схема: подписчик события не мутирует поля, а **пересобирает весь frozen
dataclass целиком** и атомарно подменяет ссылку в holder'е:

```python
# до (мутация на лету):
setattr(settings, "rerank_min_score", 0.2)

# после (новый неизменяемый объект, подмена ссылки):
holder.rag = build_rag_settings(raw_settings)  # одно присваивание атрибута — атомарно в CPython
```

Свойства схемы:

- запрос, уже взявший снапшот, работает со старым объектом до конца —
  согласованность гарантирует `frozen=True`. Эта семантика уже заявлена в
  докстринге `domain/value_objects/rag_settings.py` («snapshot remains stable») —
  сейчас она держится на честном слове, станет механической гарантией;
- новый запрос получает новый объект;
- никаких локов, никакого состояния «наполовину обновлено»;
- цена: подписчик обязан знать, как пересобрать подконфиг — переиспользуется та же
  builder-функция, что в composition root.

### D4. Миграция — постепенная, фасад + ruff-запрет новых импортов

| Вариант | Плюсы | Минусы | Вердикт |
|---|---|---|---|
| A. Big bang, один PR | Быстро | ~40 файлов; `rag_service.py` — god-file (928 строк), по правилам AGENTS.md требует characterization tests ДО рефакторинга; PR невозможно ревьюить; нет точек отката | ❌ |
| **B. Фасад + постепенная миграция по подсистемам** | Каждый шаг мал, тестируем и откатываем изолированно; ruff `banned-api` не пускает новые `from config import settings` | Дольше; переходный период с двумя механизмами | ✅ |

Механика: старый `settings`-синглтон остаётся работать, пока подсистемы одна за
другой переводятся на подконфиги. Ruff-правило `flake8-tidy-imports.banned-api`
запрещает **новые** импорты `config.settings` везде, кроме `composition/**`;
легаси-файлы вносятся в `per-file-ignores`, и этот список только сокращается.
Без ruff-запрета список легаси-файлов начал бы расти и миграция «протухла» бы.

### D5. DI — composition root, конструкторная инъекция

Worker и CLI работают вне FastAPI-контекста — `Depends()` не покрывает их.
Используется существующий самописный контейнер `server/app/composition/**`.
Подконфиг — обычный аргумент конструктора, без service locator.

### D6. Тестируемость — frozen dataclasses inline + фабрики `make_*_settings(**overrides)`

Паттерн уже есть в `tests/fakes.py` — расширяем, не изобретаем. Тест создаёт
подконфиг одной строкой, без env-переменных и без полного `Settings`.

---

## Целевая архитектура

```
.env → config.Settings (flat, ПРИВАТНЫЙ загрузчик + валидация секретов)
              │
              ▼  builder-функции (единственное место связывания)
       frozen dataclass подконфиги:
         DatabaseSettings    RedisSettings       StorageSettings(s3)
         LLMProviderSettings EmbeddingSettings  AuthSettings(jwt/admin)
         RagSettings (уже есть)  IngestionSettings(ocr/chunk)
         WorkerSettings      HttpServerSettings(cors/proxy/tz/meta)
              │
              ▼  composition root (конструкторная инъекция)
       сервисы видят ТОЛЬКО свой подконфиг
```

Правила:

- `from config import settings` разрешён ТОЛЬКО в `config.py` и `composition/**`
  (плюс `infrastructure/ml/config/settings_adapters.py` на переходный период);
- подконфиги — `@dataclass(frozen=True)`; для hot-reloadable полей — atomic swap
  всего подконфига по `ConfigParameterChanged`, не `setattr` по полям;
- `DYNAMIC_PARAMS` (domain) остаётся единым реестром hot-reloadable параметров;
- формат `.env` НЕ меняется (docker-compose, Taskfile, `.env.example` не трогаем).

---

## Этапы

### Phase 0 — Quick wins (без архитектурных изменений)

1. Объявить `ocr_min_confidence: float` в Settings (P5).
2. Исправить типы: `db_port: int`, `ocr_lang_surya: list[str]` (P7).
   `rerank_min_score: float | None` НЕ трогать — `None` означает «порог выключен».
3. Тест-реестр: каждое имя из DYNAMIC_PARAMS существует в Settings, и каждый
   builder покрывает все поля своего подконфига (анти-опечатки, сверка трёх
   сущностей: DYNAMIC_PARAMS ↔ Settings ↔ подконфиги).
4. Тест/фикс P6: убрать `s3_access_key`/`s3_secret_key` из DYNAMIC_PARAMS
   (секреты не hot-reload; ротация кредов = restart — задокументировать).

### Phase 1 — Выделение RAG-конфига (главная цель)

1. **Characterization tests на `RagService.stream()`** — обязательны по правилу
   god-files (rag_service.py, 928 строк). Baseline зелёный ДО рефакторинга.
2. Существующие `RagSettingsPort` + `LiveRagSettings` принимаются как интерфейс;
   `LiveRagSettings` переводится с чтения global settings на holder с atomic swap
   (D3). Потребитель (rag_service) не меняется — меняется только источник под
   адаптером.
3. Убрать `from config import settings` из `infrastructure/ml/rag/**`
   (rag_steps, rag_relevance, rag_retrieval) — только порт + переданный подконфиг.
4. LLM-параметры (`llm_model`, `llm_provider`, ctx-бюджеты) → отдельный
   `LLMProviderSettings`, инжектируется туда же.
5. Проверка: в `infrastructure/ml/rag/**` нет обращений к
   `s3_*`/`jwt_*`/`db_*`/`admin_*`.

### Phase 2 — Устранение P2 (application layer violations)

1. `ingestion_orchestrator.py`: config + 7 infrastructure-импортов → через порты
   (`application/ports/`), подконфиги через конструктор.
2. `benchmark_orchestrator.py`, `retrieval.py` — аналогично.
3. `grep -rn "^from infrastructure" server/app/application/` → пусто.

### Phase 3 — Остальные подсистемы

По одной подсистеме за PR: Storage → Auth → Database/Redis → Ingestion(OCR)
→ Worker → presentation routes (`documents.py`, `admin_quality.py`, CLI).

Каждый PR: builder + frozen dataclass + миграция потребителей + тест.
Порядок по частоте изменений (Storage первым — механизм кэш-инвалидации уже есть).

### Phase 4 — Удаление фасада

1. Убрать легаси-адаптеры (`settings_adapters.py` → пусто).
2. Ruff `flake8-tidy-imports.banned-api`: `config.settings` запрещён везде,
   `per-file-ignores` сокращены до нуля.
3. Решить судьбу `get_setting()`/`_settings_overrides` (benchmark ContextVar
   overrides) — либо остаётся в `config.py` как scoped-override механизм для
   benchmark, либо переезжает в benchmark-подсистему.

---

## Definition of Done

- `grep -rln "from config import settings" server/app/` → только `config.py`,
  `composition/**` (адаптеры удалены).
- `grep -rn "s3_\|jwt_\|db_password\|admin_password" server/app/infrastructure/ml/rag/` → пусто.
- `grep -rn "^from infrastructure" server/app/application/` → пусто.
- `task test`, `task lint` зелёные; characterization-тесты Phase 1 зелёные до и
  после рефакторинга.
- Hot-reload: тест на atomic swap — смена параметра mid-request не ломает снапшот
  текущего запроса (семантика `RagSettings` сохранена).

---

## Риски и митигации

| Риск | Митигация |
|------|-----------|
| Регрессия `RagService.stream()` (god-file) | Characterization tests ДО рефакторинга; сам `stream()` в этом плане НЕ переписывается — меняется только источник конфига |
| Разъезд DYNAMIC_PARAMS и подконфигов | Тест-реестр Phase 0.3; subscriber собирает подконфиг из builder'а, а не вручную |
| Секреты в hot-reload пути (P6) | Убрать секреты из DYNAMIC_PARAMS; ротация кредов = restart (документировать) |
| Протухание миграции (новые импорты синглтона в легаси-файлы) | Ruff `banned-api` с первого дня; `per-file-ignores` только сокращается |
| Объём Phase 3 | Одна подсистема = один PR; порядок по частоте изменений |

---

## Вне объёма (explicit non-goals)

- Переписывание `RagService.stream()` (отдельный пункт в refactor backlog).
- Смена формата `.env` на nested (`env_nested_delimiter`).
- Перенос статических полей в DB-dynamic config.
- Введение сторонних DI-библиотек.
