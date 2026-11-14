"""Check Compose connection budgets, including one replacement replica."""

import json
import sys


def connection_budget(services: dict, postgres_limit: int) -> dict[str, int]:
    clients = direct = surge_clients = surge_direct = 0
    for role in ("server", "worker", "worker-batch", "maintenance"):
        service = services.get(role, {})
        env = service.get("environment", {})
        replicas = int(service.get("deploy", {}).get("replicas", 1))
        pools = 1 + len([host for host in str(env.get("DB_SLAVE_HOSTS", "")).split(",") if host.strip()])
        per_replica = pools * (int(env.get("DB_POOL_SIZE", 5)) + int(env.get("DB_MAX_OVERFLOW", 0)))
        # Dedicated LISTEN plus bounded workers holding session locks. Leave
        # additional room below for admin tools and request-level direct locks.
        locks = {"worker": "DOCUMENT_MAX_CONCURRENT", "worker-batch": "WORKER_MAX_CONCURRENT", "maintenance": "OUTBOX_MAX_CONCURRENT"}
        direct_per_replica = 1 + (int(env.get(locks[role], 5 if role == "maintenance" else 4)) if role in locks else 0)
        if role == "maintenance":
            direct_per_replica += 1  # outbox LISTEN
        clients += replicas * per_replica
        direct += replicas * direct_per_replica
        surge_clients = max(surge_clients, per_replica)
        surge_direct = max(surge_direct, direct_per_replica)
    pool = services.get("pgbouncer", {}).get("environment", {})
    frontend = int(pool.get("MAX_CLIENT_CONN", 1000))
    backend = int(pool.get("MAX_DB_CONNECTIONS", 60))
    reserve = 30
    if clients + surge_clients > frontend:
        raise ValueError("Replica pools plus rolling surge exceed PgBouncer MAX_CLIENT_CONN")
    if backend + direct + surge_direct + reserve > postgres_limit:
        raise ValueError("PgBouncer backends plus direct connections and reserve exceed Postgres max_connections")
    return {"application_clients_with_surge": clients + surge_clients,
            "pgbouncer_client_limit": frontend, "pgbouncer_backend_limit": backend,
            "direct_connections_with_surge": direct + surge_direct,
            "postgres_reserve": reserve, "postgres_max_connections": postgres_limit}


if __name__ == "__main__":
    try:
        print(json.dumps(connection_budget(json.load(sys.stdin)["services"], int(sys.argv[1]))))
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Connection budget rejected: {exc}") from exc
