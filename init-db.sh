#!/usr/bin/env bash
set -euo pipefail

echo "[init] Iniciando restauración…"

# Asegura que exista la BD destino
createdb -U "$POSTGRES_USER" "$POSTGRES_DB" || true

if [ -f /docker-entrypoint-initdb.d/backup.dump ]; then
  echo "[init] Encontrado backup.dump, restaurando con pg_restore…"
  # --clean/--if-exists eliminan objetos antes de recrearlos
  # -j paraleliza si el dump fue generado con -Fd (dir); si es -Fc, deja -j en 4 igual
  pg_restore \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --clean --if-exists \
    -j 4 \
    /docker-entrypoint-initdb.d/backup.dump

  echo "[init] Restauración finalizada."
else
  echo "[init][WARN] No se encontró /docker-entrypoint-initdb.d/backup.dump"
fi

# Habilita la extensión vector (si no vino en el dump)
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"
