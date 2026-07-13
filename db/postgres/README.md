# db/postgres — схема и инициализация БД истории диалогов

Каталог смонтирован в контейнер `postgres` двумя volume'ами (см. `docker-compose.yml`):

```
db/postgres/init/  →  /docker-entrypoint-initdb.d/   (только при первом запуске на пустом volume)
pg_data/            →  /var/lib/postgresql/data       (данные, создаётся автоматически)
```

## init/ — инициализация схемы

Официальный образ `postgres` при **первом** старте (когда `/var/lib/postgresql/data` ещё пуст)
выполняет все `*.sql` / `*.sh` файлы из `/docker-entrypoint-initdb.d/` **по алфавиту**. Поэтому
файлы пронумерованы:

```
init/
└── 01_schema.sql   ← таблицы conversations, messages + индексы
```

Если понадобится расширить схему — добавляй новый файл с следующим номером
(`02_add_feedback_table.sql` и т.п.), не редактируй `01_schema.sql` задним числом: он уже
применился на всех существующих окружениях, где `pg_data/` не пуст.

## Что делать, если схему нужно поменять на уже существующей БД

`init/` выполняется только на пустом volume. Для боевой базы с данными — либо:

```bash
# накатить новый SQL-файл вручную
docker exec -i rag_postgres psql -U raguser -d ragdb < db/postgres/init/02_*.sql

# либо (для локальной разработки, если не жалко данных диалогов) — пересоздать volume
docker compose down
rm -rf pg_data/
docker compose up -d postgres
```

## Резервное копирование

```bash
docker exec rag_postgres pg_dump -U raguser ragdb > backup_$(date +%F).sql
```
