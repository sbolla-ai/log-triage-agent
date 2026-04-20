# Database Troubleshooting Runbook

## Symptom: Connection timeouts
1. Check if the database host is reachable: `ping db.internal`
2. Verify the connection pool isn't exhausted: query `pg_stat_activity`
3. Check if any long-running queries are blocking: `SELECT * FROM pg_locks;`
4. If timeouts persist, restart the connection pool service

## Symptom: Slow queries
1. Enable slow query logging: set `log_min_duration_statement = 1000`
2. Look for missing indexes via `EXPLAIN ANALYZE`
3. Consider VACUUM ANALYZE on affected tables

## Escalation
Page the database on-call if errors persist longer than 5 minutes.
