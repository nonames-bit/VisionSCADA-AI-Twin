"""Creates the 'mlflow' database if missing (works on existing Postgres volumes) and starts MLflow."""
import os
import time

import psycopg2

host = os.getenv("PG_HOST", "postgres-vision")
user = os.getenv("PG_ADMIN_USER", "twin")
password = os.getenv("PG_ADMIN_PASS", "change_me")
db = os.getenv("MLFLOW_DB", "mlflow")

for attempt in range(30):
    try:
        conn = psycopg2.connect(host=host, user=user, password=password, dbname="postgres")
        break
    except Exception as e:
        print(f"waiting for postgres ({e})", flush=True)
        time.sleep(2)
else:
    raise SystemExit("postgres not reachable")
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{db}"')
        print(f"created database {db}", flush=True)
conn.close()

os.execvp(
    "mlflow",
    [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", f"postgresql://{user}:{password}@{host}:5432/{db}",
        "--serve-artifacts",
        "--artifacts-destination", os.getenv("MLFLOW_ARTIFACTS_DESTINATION", "s3://mlflow"),
    ],
)
