# Production Rollout Guide

## Prerequisites

1. External stateful infrastructure provisioned:
   - Managed PostgreSQL (CloudSQL / RDS / Azure Database)
   - Managed Redis (ElastiCache / Memorystore / Azure Cache)
   - Qdrant Cloud or self-hosted cluster
   - S3-compatible object storage
   - TEI embedding/reranking endpoints

2. Secrets management configured:
   - External Secrets Operator or Sealed Secrets
   - All production secrets injected from vault, NOT from Git

3. DNS and TLS:
   - Domain pointed to Ingress controller
   - cert-manager configured with ClusterIssuer

## Deployment Steps

### 1. Pre-flight checks

```bash
# Verify cluster connectivity
kubectl cluster-info

# Verify secrets exist
kubectl get secrets -n rag-prod

# Verify external services are reachable from cluster
kubectl run --rm -it debug --image=curlimages/curl -- \
  curl -s https://managed-pg-host:5432 || echo "DB unreachable"
```

### 2. First deploy

```bash
# Install/upgrade Helm release
helm upgrade --install rag-app ./deploy/helm/rag-app \
  -f ./deploy/helm/rag-app/values-prod.yaml \
  -n rag-prod \
  --create-namespace \
  --set secrets.jwtSecretKey="$(vault read -field=jwt_secret secret/rag)" \
  --set secrets.dbPassword="$(vault read -field=db_password secret/rag)" \
  --set secrets.qdrantApiKey="$(vault read -field=qdrant_api_key secret/rag)" \
  --wait --timeout 600s

# Verify health
kubectl get pods -n rag-prod -l "app.kubernetes.io/component=server"
curl -s https://rag.example.com/health | jq .
```

### 3. Rollback procedure

```bash
# List available revisions
helm history rag-app -n rag-prod

# Rollback to previous revision
helm rollback rag-app <REVISION> -n rag-prod

# Or rollback to specific revision
helm rollback rag-app 3 -n rag-prod

# Verify rollback
kubectl rollout status deployment/rag-app-server -n rag-prod
curl -s https://rag.example.com/health | jq .
```

### 4. Scaling

```bash
# Manual scale (temporary, HPA will override)
kubectl scale deployment/rag-app-server --replicas=5 -n rag-prod

# Check HPA
kubectl get hpa -n rag-prod
kubectl describe hpa rag-app-server -n rag-prod
```

### 5. Monitoring post-deploy

```bash
# Watch pods
kubectl get pods -n rag-prod -w -l "app.kubernetes.io/component=server"

# Check logs
kubectl logs -n rag-prod -l "app.kubernetes.io/component=server" --tail=100

# Check for OOM
kubectl describe pods -n rag-prod -l "app.kubernetes.io/component=server" | grep -A5 "Last State"

# Grafana dashboard
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Open: http://localhost:3000 (admin/admin)
```

## Acceptance Criteria Verification

| # | Criteria | Verification |
|---|----------|--------------|
| 1 | App runs in k3d | `kubectl get pods` shows Running |
| 2 | `tilt up` works | `tilt up` builds and deploys |
| 3 | API via Helm Deployment | `helm status rag-app` |
| 4 | Worker separate Deployment | `kubectl get deploy rag-app-worker` |
| 5 | DB/Redis/Qdrant accessible | `curl /health` shows all OK |
| 6 | Migrations via Job | `kubectl get jobs` shows completed |
| 7 | No migrations in API pods | entrypoint.sh has no alembic step |
| 8 | Secrets not in ConfigMap | `kubectl get cm -o yaml` has no secrets |
| 9 | Readiness/liveness probes | `kubectl describe pod` shows probes |
| 10 | HPA configured | `kubectl get hpa` shows targets |
| 11 | PDB configured | `kubectl get pdb` |
| 12 | 3+ replicas work | `kubectl scale --replicas=3` + health OK |
| 13 | SSE works via Ingress | Test streaming with `curl -N` |
| 14 | No Ingress buffering | SSE tokens arrive immediately |
| 15 | Prod stateful external | `values-prod.yaml` has `external: true` |
| 16 | LogBufferHandler not attached | main.py has no attach_log_buffer call |
| 17 | Logs centralized via Loki | Loki/Grafana shows all pod logs |
| 18 | /metrics collected by Prometheus | Prometheus targets page |
| 19 | Grafana dashboards | Dashboard shows API metrics |
| 20 | Load test passes | k6 results show no errors |
| 21 | HPA scales under load | Replicas increase during load test |
| 22 | Reproducible deployment | `helm upgrade --install` is idempotent |
| 23 | Rollback works | `helm rollback` restores previous version |

## CI/CD Pipeline (GitOps)

```
git push
  → CI (lint, test, build)
  → Build Docker image (tag = commit SHA)
  → Push to registry
  → Update Helm values with new image tag
  → helm upgrade --install (staging auto, prod manual approval)
```

For GitOps with ArgoCD/FluxCD:
- Store Helm values in Git (secrets via External Secrets Operator)
- ArgoCD watches Git repo and syncs to cluster
- Rollback = revert Git commit
