# Kubernetes Migration — Полная инструкция

## Содержание

1. [Предварительные требования](#1-предварительные-требования)
2. [Локальная разработка (k3d)](#2-локальная-разработка-k3d)
3. [Ручной деплой через Helm](#3-ручной-деплой-через-helm)
4. [Observability (метрики, логи, дашборды)](#4-observability)
5. [Нагрузочное тестирование](#5-нагрузочное-тестирование)
6. [Production деплой](#6-production-деплой)
7. [Rollback](#7-rollback)
8. [Справка по командам](#8-справка-по-командам)
9. [Чеклист acceptance criteria](#9-чеклист-acceptance-criteria)

---

## 1. Предварительные требования

Установить на хост-машине:

```bash
# k3d — локальный Kubernetes через Docker
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Helm — пакетный менеджер для Kubernetes
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# kubectl — CLI для Kubernetes
curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# task — task runner (Taskfile.yml)
curl -s https://taskfile.dev/install.sh | sh && sudo mv bin/task /usr/local/bin/

# k6 (опционально) — нагрузочное тестирование
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
  --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D68
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
  sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update && sudo apt-get install k6 -y
```

Проверить версии:

```bash
k3d version
helm version
kubectl version --client
```

---

## 2. Локальная разработка (k3d)

### 2.1. Быстрый старт

```bash
task k3d:create       # Создать k3d кластер
task helm:infra       # Поднять инфру (postgres, redis, qdrant, ollama, minio, ingress)
task helm:upgrade     # Задеплоить приложение
```

После этого:
- API доступен через port-forward на `http://localhost:8001`
- k3d cluster context: `k3d-rag-dev`

### 2.2. Пошаговый процесс

**Шаг 1 — Создать k3d кластер**

```bash
task k3d:create
```

Что происходит:
- Запускается k3d кластер `rag-dev` с 1 серверным нодом
- Настраивается registry `registry.localhost:5000` для локальных образов
- Открываются порты: 8080 (HTTP), 8443 (HTTPS)
- kubectl настроен на новый кластер (`k3d-rag-dev`)

Проверить:
```bash
kubectl get nodes
# NAME                     STATUS   ROLES                  AGE   VERSION
# k3d-rag-dev-server-0     Ready    control-plane,master   30s   v1.x.x
```

**Шаг 2 — Поднять инфраструктуру**

```bash
task helm:infra
```

Что происходит (через OCI-registry Bitnami + traditional repos):

| Сервис | Источник чарта |
|--------|---------------|
| PostgreSQL | `oci://registry-1.docker.io/bitnamicharts/postgresql` |
| Redis | `oci://registry-1.docker.io/bitnamicharts/redis` |
| Qdrant | `qdrant/qdrant` (repo: `https://qdrant.github.io/qdrant-helm`) |
| Ollama | `otwld/ollama` (repo: `https://otwld.github.io/helm-charts`) |
| MinIO | `oci://registry-1.docker.io/bitnamicharts/minio` |
| Ingress NGINX | `ingress-nginx/ingress-nginx` (repo: `https://kubernetes.github.io/ingress-nginx`) |

> **Важно:** Bitnami перешёл на OCI (ноябрь 2024). Старый URL `charts.bitnami.com` возвращает 403.
> Qdrant переехал: `qdrant.github.io/helm-charts` → `qdrant.github.io/qdrant-helm`.

Проверить:
```bash
kubectl get pods
# NAME                                                READY   STATUS    RESTARTS   AGE
# rag-app-postgres-postgresql-0                       1/1     Running   0          2m
# rag-app-redis-master-0                              1/1     Running   0          2m
# rag-app-qdrant-0                                    1/1     Running   0          2m
# rag-app-ollama-xxx                                  1/1     Running   0          2m
# rag-app-ingress-nginx-controller-xxx                1/1     Running   0          2m
```

> **DNS-имена Bitnami добавляют суффиксы:**
> - `rag-app-postgres` → `rag-app-postgres-postgresql`
> - `rag-app-redis` → `rag-app-redis-master`
> Уже исправлено в `values.yaml` / `values-dev.yaml`.

**Шаг 3 — Собрать и загрузить образ**

```bash
# Собрать (если ещё не собран)
task build

# Загрузить в локальный registry k3d
docker tag rag-server:latest registry.localhost:5000/rag-server:latest
docker push registry.localhost:5000/rag-server:latest
```

**Шаг 4 — Задеплоить приложение**

```bash
task helm:upgrade
```

Это выполняет:
```bash
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-dev.yaml \
  --set image.repository=registry.localhost:5000/rag-server \
  --set image.tag=latest \
  --set image.pullPolicy=Always \
  --set migrations.enabled=false
```

> **Миграции отключены** (`migrations.enabled=false`) из-за бага: migration Job — это `pre-upgrade` hook, который запускается до создания ConfigMap/Secret, но сам depends на них через `envFrom`. Исправление чарта TODO.

**Шаг 5 — Port-forward и проверка**

```bash
# Вкладка 1 — port-forward (висит, не закрывай)
kubectl port-forward svc/rag-app-server 8001:80

# Вкладка 2 — health check
curl -s http://localhost:8001/health | jq .

# Вкладка 3 — логи
kubectl logs -l "app.kubernetes.io/component=server" --tail=30 -f
```

> **k3d LoadBalancer не проксирует на NodePort.** Доступ к API только через port-forward. `rag.local` не работает без дополнительной настройки DNS.

**Шаг 6 — Клиент (Web UI)**

Клиент не задеплоен в k3d. Запускай локально:

```bash
cd client
VITE_API_URL=http://localhost:8001 npm run dev
```

Открой `http://localhost:5173` в браузере.

### 2.3. Остановка

```bash
task k3d:delete    # Удалить k3d кластер полностью
```

---

## 3. Ручной деплой через Helm

### 3.1. Просмотр манифестов

```bash
task helm:template
```

### 3.2. Деплой по окружениям

```bash
# Dev (k3d, in-cluster сервисы)
task helm:upgrade

# Staging (2 реплики, HPA)
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-staging.yaml

# Production (3+ реплик, внешние сервисы, TLS)
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-prod.yaml \
  -n rag-prod --create-namespace
```

### 3.3. Переопределение значений

```bash
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-dev.yaml \
  --set server.replicas=2 \
  --set config.logLevel=DEBUG
```

### 3.4. Просмотр текущего статуса

```bash
task helm:status
```

---

## 4. Observability

### 4.1. Развернуть стек

```bash
task observability:up
```

Устанавливает в namespace `monitoring`:
- **Prometheus** — сбор метрик
- **Grafana** — дашборды
- **Loki + Promtail** — централизованный сбор логов

### 4.2. Доступ к Grafana

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# http://localhost:3000 (admin / admin)
```

### 4.3. Доступ к Prometheus

```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# http://localhost:9090
```

### 4.4. Логи через Loki

В Grafana: Add data source → Loki → URL: `http://loki:3100`

Запросы:
```
{app="rag-app", component="server"}
{app="rag-app"} | logfmt | level="error"
{app="rag-app", pod=~"rag-app-server.*"}
```

### 4.5. Логи без Grafana

```bash
kubectl logs -l app.kubernetes.io/component=server -f --tail=100
kubectl logs -l app.kubernetes.io/component=server --tail=100 | grep -i error
kubectl logs -l app.kubernetes.io/component=worker -f --tail=100
```

### 4.6. Удалить стек observability

```bash
task observability:down
```

---

## 5. Нагрузочное тестирование

```bash
# Smoke test
task loadtest:smoke

# Полный нагрузочный тест
task loadtest:run

# Spike test
task loadtest:spike

# Soak test (поиск утечек памяти)
task loadtest:soak

# SSE streaming test
task loadtest:sse
```

---

## 6. Production деплой

### 6.1. Подготовка

1. Внешние сервисы: Managed PostgreSQL, Redis, Qdrant Cloud, S3, TEI endpoints
2. Secrets management: External Secrets Operator / Sealed Secrets / Vault
3. DNS + TLS: cert-manager с ClusterIssuer

### 6.2. Деплой

```bash
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-prod.yaml \
  -n rag-prod \
  --create-namespace \
  --wait --timeout 600s

curl https://rag.example.com/health | jq .
```

### 6.3. Secrets в production

```yaml
# Вариант A — External Secrets Operator + Vault
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: rag-app-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: rag-app-secret
  data:
    - secretKey: JWT_SECRET_KEY
      remoteRef:
        key: secret/rag
        property: jwt_secret
    - secretKey: DB_PASSWORD
      remoteRef:
        key: secret/rag
        property: db_password
```

```bash
# Вариант B — helm --set
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-prod.yaml \
  --set secrets.jwtSecretKey="$(vault read -field=jwt_secret secret/rag)" \
  --set secrets.dbPassword="$(vault read -field=db_password secret/rag)"
```

---

## 7. Rollback

```bash
# История релиза
helm history rag-app -n rag-prod

# Откатить
helm rollback rag-app <REVISION> -n rag-prod

# Проверить
kubectl rollout status deployment/rag-app-server -n rag-prod
curl https://rag.example.com/health | jq .
```

---

## 8. Справка по командам

### Taskfile

| Команда | Описание |
|---------|----------|
| `task k3d:create` | Создать k3d кластер |
| `task k3d:delete` | Удалить k3d кластер |
| `task k3d:list` | Показать k3d кластеры |
| `task helm:infra` | Поднять инфру в k3d |
| `task helm:upgrade` | Задеплоить приложение |
| `task helm:template` | Просмотр манифестов |
| `task helm:status` | Статус Helm-релиза |
| `task helm:rollback` | Откатить Helm-релиз |
| `task k8s:logs` | Логи API |
| `task k8s:logs:worker` | Логи worker |
| `task k8s:describe` | Описание pod'ов |
| `task k8s:port-forward` | Port-forward к API |
| `task k8s:migrate` | Миграции вручную |
| `task k8s:scale REPLICAS=3` | Масштабировать API |
| `task k8s:status` | Статус всех ресурсов |
| `task build` | Собрать Docker образ |
| `task up` | Поднять Docker Compose стек |
| `task down` | Остановить Docker Compose стек |
| `task observability:up` | Prometheus/Grafana/Loki |
| `task observability:down` | Удалить observability |

### kubectl

```bash
# Pod'ы
kubectl get pods -l app.kubernetes.io/instance=rag-app
kubectl describe pod <pod-name>
kubectl logs <pod-name> -f
kubectl exec -it <pod-name> -- bash

# Масштабирование
kubectl scale deployment/rag-app-server --replicas=5

# HPA / PDB
kubectl get hpa
kubectl get pdb

# Ingress
kubectl get ingress

# Secrets
kubectl get secret rag-app-secret -o yaml

# Все ресурсы приложения
kubectl get all,ingress,configmap,secret,job -l "app.kubernetes.io/instance=rag-app"
```

### Доступ к API

```bash
# Port-forward
kubectl port-forward svc/rag-app-server 8001:80

# Health
curl -s http://localhost:8001/health | jq .

# Авторизация
curl -sX POST http://localhost:8001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "..."}'

# Клиент (Web UI)
cd client && VITE_API_URL=http://localhost:8001 npm run dev
```

---

## 9. Чеклист acceptance criteria

```
[ ] Приложение запускается в k3d
[ ] API работает через Helm Deployment
[ ] worker работает отдельным Deployment
[ ] PostgreSQL/Redis/Qdrant доступны из API
[ ] secrets не хранятся в ConfigMap
[ ] настроены readiness/liveness probes (path: /health)
[ ] настроен HPA
[ ] настроен PDB
[ ] API корректно работает минимум с 3 replicas
[ ] SSE корректно работает через Kubernetes Ingress
[ ] production stateful-компоненты вынесены за пределы Kubernetes
[ ] логи доступны через kubectl logs
[ ] /metrics собирается Prometheus
[ ] Grafana содержит основные метрики
[ ] существует воспроизводимый deployment
[ ] существует rollback-сценарий
```
