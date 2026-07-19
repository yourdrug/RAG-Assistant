-- Пользователи. Саморегистрации нет намеренно — это закрытый инструмент компании.
-- Первый admin создаётся автоматически при старте приложения (см. main.py:bootstrap_admin,
-- переменные ADMIN_EMAIL/ADMIN_PASSWORD). Дальше новых пользователей заводит сам admin
-- через POST /auth/users.
--
-- kind разделяет внутренних сотрудников и внешних клиентов — это отдельная ось
-- от role (admin/user). Клиент не может быть admin (см. CHECK ниже).
CREATE TABLE IF NOT EXISTS users (
    id               SERIAL PRIMARY KEY,
    email            VARCHAR(255) UNIQUE NOT NULL,
    hashed_password  VARCHAR(255) NOT NULL,
    role             VARCHAR(16) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    kind             VARCHAR(16) NOT NULL DEFAULT 'internal' CHECK (kind IN ('internal', 'client')),
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMP DEFAULT NOW(),
    CONSTRAINT chk_client_not_admin CHECK (NOT (kind = 'client' AND role = 'admin'))
);

-- Таблица диалогов — привязана к конкретному пользователю
CREATE TABLE IF NOT EXISTS conversations (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Таблица сообщений
CREATE TABLE IF NOT EXISTS messages (
    id               SERIAL PRIMARY KEY,
    conversation_id  INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role             VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    sources          JSONB,          -- источники документов, найденные RAG
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);

CREATE TABLE IF NOT EXISTS groups (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS user_groups (
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, group_id)
);

CREATE TABLE IF NOT EXISTS client_assignments (
    internal_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    client_user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assigned_by        INTEGER REFERENCES users(id),
    assigned_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (internal_user_id, client_user_id)
);

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    filename      VARCHAR(255) NOT NULL,
    source_path   TEXT NOT NULL DEFAULT '',
    visibility    VARCHAR(20) NOT NULL
                  CHECK (visibility IN ('internal_public', 'internal_group', 'internal_private', 'client_private')),
    owner_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    group_id      INTEGER REFERENCES groups(id) ON DELETE CASCADE,
    status        VARCHAR(16) NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    error_message TEXT,
    chunks        INTEGER,
    chars         INTEGER,
    created_at    TIMESTAMP DEFAULT NOW(),
    indexed_at    TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_active_slot
    ON documents (owner_id, filename, COALESCE(group_id, -1))
    WHERE status IN ('pending', 'processing', 'done');

CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_group ON documents(group_id);
CREATE INDEX IF NOT EXISTS idx_documents_visibility ON documents(visibility);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
