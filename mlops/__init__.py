"""
MLOps layer for VisionSCADA AI Twin.

- catalog:    recipes, honest class labels and storage paths
- datasets:   labeled image store (storage/datasets), content-hash train/test/holdout split
- drift:      real Population Stability Index (PSI) against the training reference
- registry:   local champion/challenger model registry (models/registry.json)
- tracking:   optional MLflow experiment tracking + Model Registry aliases
- versioning: optional DVC snapshot of the dataset (S3 remote on MinIO)
- trainer:    feature extraction + training + evaluation + registration
- jobs:       background retraining job manager (triggered from Ignition / 3D twin)
"""
