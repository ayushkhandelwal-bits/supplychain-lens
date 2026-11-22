# ADR-0002 — SQLite instead of PostgreSQL; no Kafka, Redis or Celery

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

NEXUS-SCCT's docker-compose runs 14 services: PostgreSQL, Redis, Zookeeper, Kafka, Kafka UI, MLflow, API, Celery worker, Celery beat, frontend, Prometheus, Grafana, NGINX and a Kafka setup job. Free hosting (Streamlit Community Cloud) runs one Python process with limited memory; it can't host any of these as separate services.

The workload is also small: about 150k demand rows and a few hundred rows in every other table. The data changes only when the generator runs.

## Decision

| Original | Replacement | Why it's enough here |
|---|---|---|
| PostgreSQL | **SQLite** file built by the pipeline | Single-file, zero-config, SQL-compatible; read-mostly workload of this size is trivial for it |
| Redis cache | `st.cache_data` / `st.cache_resource` + precomputed artifacts | Single process; nothing to share across machines |
| Kafka event streaming | An `events` table and a "replay" control that steps through events in time order | Shows the event-driven *concept* without a broker |
| Celery workers + beat | `pipelines/build_all.py` run offline (optionally by GitHub Actions on a schedule) | Retraining doesn't need to happen inside the web app |
| WebSockets / pub-sub | Streamlit reruns | Demo doesn't need push updates |

## Consequences

- We lose real-time multi-user streaming. That's acceptable for a portfolio demo, and the README will say so.
- The query layer will use plain SQL with parameters, so moving to PostgreSQL later is mostly a connection-string change.
- Interview framing: *"I right-sized the infrastructure to the workload and the budget, and kept the seams so it can scale back up."*
