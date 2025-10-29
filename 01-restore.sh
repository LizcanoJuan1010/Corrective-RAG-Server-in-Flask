#!/bin/sh
set -eu

echo "[init] Esperando Postgres…"
until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
  sleep 1
done

# Por si acaso:
createdb -U "$POSTGRES_USER" "$POSTGRES_DB" 2>/dev/null || true

echo "[init] Habilitando extensión vector…"
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "[init] Restaurando backup.dump con pg_restore…"
pg_restore \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  --no-owner --no-privileges \
  -j 4 \
  /docker-entrypoint-initdb.d/backup.dump

echo "[init] ¡Restauración finalizada!"
