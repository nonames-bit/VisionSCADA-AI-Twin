# MLflow tracking server + Model Registry
#   backend store : PostgreSQL (database "mlflow" in the existing TimescaleDB container)
#   artifacts     : MinIO (S3) bucket "mlflow", proxied by the server (--serve-artifacts),
#                   so clients (server.py, Ignition) only need HTTP, not S3 credentials.
FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3
COPY entrypoint.py /entrypoint.py
ENTRYPOINT ["python", "/entrypoint.py"]
