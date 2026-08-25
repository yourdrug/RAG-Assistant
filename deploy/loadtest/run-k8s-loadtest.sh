#!/bin/bash
# Kubernetes load test runner
# Validates: HPA scaling, multi-replica SSE, rolling updates, log aggregation
#
# Prerequisites:
#   - k3d cluster running (task k3d:create)
#   - App deployed via Helm (task helm:upgrade)
#   - k6 installed locally
#
# Usage: ./deploy/loadtest/run-k8s-loadtest.sh [scenario]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
NAMESPACE="${NAMESPACE:-default}"
DURATION="${DURATION:-5m}"
VUS="${VUS:-50}"
SCENARIO="${1:-scaling}"

mkdir -p "${RESULTS_DIR}"

echo "============================================="
echo "RAG Kubernetes Load Test — Scenario: ${SCENARIO}"
echo "============================================="

# Get current pod count before test
echo ""
echo "[Pre-test] Current API replicas:"
kubectl get deployment rag-app-server -n "${NAMESPACE}" -o custom-columns=\
"REPLICAS:.status.replicas,\
READY:.status.readyReplicas,\
UPDATED:.status.updatedReplicas,\
AVAILABLE:.status.availableReplicas"

echo ""
echo "[Pre-test] HPA status:"
kubectl get hpa rag-app-server -n "${NAMESPACE}" 2>/dev/null || echo "  (HPA not configured)"

case "${SCENARIO}" in
  scaling)
    echo ""
    echo "Running scaling test: ${VUS} VUs for ${DURATION}"
    echo "Expected: HPA should scale from 1 -> N replicas"
    k6 run \
      --out json="${RESULTS_DIR}/scaling_${TIMESTAMP}.json" \
      -e BASE_URL="http://$(kubectl get svc rag-app-server -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')/chat" \
      -e VUS="${VUS}" \
      -e DURATION="${DURATION}" \
      "${SCRIPT_DIR}/scaling-test.js" || true
    ;;

  sse)
    echo ""
    echo "Running SSE streaming test with ${VUS} concurrent connections"
    echo "Expected: All replicas serve SSE correctly, no buffering issues"
    k6 run \
      --out json="${RESULTS_DIR}/sse_${TIMESTAMP}.json" \
      -e BASE_URL="http://$(kubectl get svc rag-app-server -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')/chat" \
      -e VUS="${VUS}" \
      "${SCRIPT_DIR}/sse-test.js" || true
    ;;

  rolling-update)
    echo ""
    echo "Running rolling update test"
    echo "Expected: Zero-downtime deployment, no failed requests"
    BEFORE_REPLICAS=$(kubectl get deployment rag-app-server -n "${NAMESPACE}" -o jsonpath='{.spec.replicas}')
    k6 run \
      --out json="${RESULTS_DIR}/rolling_${TIMESTAMP}.json" \
      -e BASE_URL="http://$(kubectl get svc rag-app-server -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')/health" \
      -e VUS=10 \
      "${SCRIPT_DIR}/availability-test.js" &
    K6_PID=$!
    sleep 5
    echo "  Triggering rolling update..."
    kubectl rollout restart deployment/rag-app-server -n "${NAMESPACE}"
    kubectl rollout status deployment/rag-app-server -n "${NAMESPACE}" --timeout=300s
    wait ${K6_PID} || true
    ;;

  *)
    echo "Unknown scenario: ${SCENARIO}"
    echo "Available: scaling, sse, rolling-update"
    exit 1
    ;;
esac

echo ""
echo "[Post-test] Final API replicas:"
kubectl get deployment rag-app-server -n "${NAMESPACE}" -o custom-columns=\
"REPLICAS:.status.replicas,\
READY:.status.readyReplicas"

echo ""
echo "[Post-test] Pod restarts:"
kubectl get pods -n "${NAMESPACE}" -l "app.kubernetes.io/component=server" \
  -o custom-columns="NAME:.metadata.name,RESTARTS:.status.containerStatuses[0].restartCount"

echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Done."
