#!/bin/bash
set -e

# Imprime un mensaje indicando que el script de restauración ha comenzado.
echo ">>>>> Iniciando la restauración de la base de datos desde el archivo de respaldo..."

# Utiliza psql para cargar los datos desde un dump de texto plano.
# El archivo backup.dump estará disponible en este directorio gracias al montaje de Docker.
# psql es la herramienta adecuada para restaurar backups de SQL en formato de texto.
# Usamos las variables de entorno estándar que la imagen de postgres provee.
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/backup.dump

# Imprime un mensaje de éxito una vez completada la restauración.
echo ">>>>> Restauración de la base de datos completada con éxito."
