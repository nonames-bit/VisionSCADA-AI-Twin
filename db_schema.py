"""
PostgreSQL Schema, Provisioning & Read-Only Copilot Guardrail Setup
for VisionSCADA AI Twin.
"""
import os
import json
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import RealDictCursor

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_ADMIN_USER = os.getenv("PG_ADMIN_USER", "twin")
PG_ADMIN_PASS = os.getenv("PG_ADMIN_PASS", "change_me")
PG_DB = os.getenv("PG_DB", "vision_scada")

PG_RO_USER = os.getenv("PG_RO_USER", "copilot_ro")
PG_RO_PASS = os.getenv("PG_RO_PASS", "copilot_readonly_2026")


def get_admin_conn(dbname: str = PG_DB):
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_ADMIN_USER,
        password=PG_ADMIN_PASS,
        dbname=dbname,
    )


def get_readonly_conn():
    """Strictly Read-Only PostgreSQL connection for the LLM Copilot."""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_RO_USER,
        password=PG_RO_PASS,
        dbname=PG_DB,
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


def init_database():
    """Creates database `vision_scada`, tables, seed suppliers/batches, and read-only role."""
    # 1. Connect to default 'twin' or 'postgres' DB to create 'vision_scada' if missing
    bootstrap_conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_ADMIN_USER,
        password=PG_ADMIN_PASS,
        dbname="twin",
    )
    bootstrap_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with bootstrap_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DB,))
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {PG_DB}")
        # Create read-only user for LLM Copilot if not exists
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (PG_RO_USER,))
        if not cur.fetchone():
            cur.execute(
                f"CREATE ROLE {PG_RO_USER} WITH LOGIN PASSWORD %s NOSUPERUSER NOCREATEDB NOCREATEROLE",
                (PG_RO_PASS,),
            )
    bootstrap_conn.close()

    # 2. Create schema inside vision_scada
    conn = get_admin_conn(PG_DB)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS suppliers (
                supplier_id SERIAL PRIMARY KEY,
                supplier_code VARCHAR(32) UNIQUE NOT NULL,
                name VARCHAR(120) NOT NULL,
                city VARCHAR(80) NOT NULL,
                country VARCHAR(80) NOT NULL,
                lat DOUBLE PRECISION NOT NULL,
                lon DOUBLE PRECISION NOT NULL,
                specialty_recipe VARCHAR(32) NOT NULL,
                baseline_defect_rate DOUBLE PRECISION DEFAULT 0.04
            );

            CREATE TABLE IF NOT EXISTS production_batches (
                batch_id SERIAL PRIMARY KEY,
                batch_code VARCHAR(40) UNIQUE NOT NULL,
                product_recipe VARCHAR(32) NOT NULL,
                product_name VARCHAR(120) NOT NULL,
                supplier_id INTEGER REFERENCES suppliers(supplier_id),
                target_qty INTEGER NOT NULL DEFAULT 500,
                inspected_qty INTEGER NOT NULL DEFAULT 0,
                ok_qty INTEGER NOT NULL DEFAULT 0,
                nok_qty INTEGER NOT NULL DEFAULT 0,
                status VARCHAR(24) NOT NULL DEFAULT 'ACTIVE',
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                model_id SERIAL PRIMARY KEY,
                product_recipe VARCHAR(32) NOT NULL,
                model_version VARCHAR(32) NOT NULL,
                deployment_role VARCHAR(24) NOT NULL, -- PRODUCTION, SHADOW, ARCHIVED
                algorithm VARCHAR(80) NOT NULL,
                accuracy DOUBLE PRECISION NOT NULL,
                precision_score DOUBLE PRECISION NOT NULL,
                recall_score DOUBLE PRECISION NOT NULL,
                f1_score DOUBLE PRECISION NOT NULL,
                roc_auc DOUBLE PRECISION NOT NULL,
                train_samples INTEGER NOT NULL,
                test_samples INTEGER NOT NULL,
                confusion_matrix_json JSONB NOT NULL,
                artifact_path VARCHAR(255) NOT NULL,
                trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(product_recipe, model_version)
            );

            CREATE TABLE IF NOT EXISTS inspections (
                inspection_id BIGSERIAL PRIMARY KEY,
                serial_number VARCHAR(48) UNIQUE NOT NULL,
                batch_id INTEGER REFERENCES production_batches(batch_id),
                supplier_id INTEGER REFERENCES suppliers(supplier_id),
                product_recipe VARCHAR(32) NOT NULL,
                image_rel_path VARCHAR(255) NOT NULL,
                heatmap_rel_path VARCHAR(255) NOT NULL,
                sha256_hash VARCHAR(64) NOT NULL,
                ground_truth_class VARCHAR(48) NOT NULL,

                -- Edge Production Model Decision (Deterministic Loop)
                edge_model_version VARCHAR(32) NOT NULL,
                edge_status VARCHAR(16) NOT NULL, -- OK, NOK
                edge_defect_class VARCHAR(48) NOT NULL,
                edge_confidence DOUBLE PRECISION NOT NULL,
                edge_anomaly_score DOUBLE PRECISION NOT NULL,
                edge_latency_ms DOUBLE PRECISION NOT NULL,
                bbox_json JSONB DEFAULT '[]'::jsonb,

                -- Shadow Model Decision (Parallel MLOps Comparison)
                shadow_model_version VARCHAR(32) NOT NULL,
                shadow_status VARCHAR(16) NOT NULL,
                shadow_defect_class VARCHAR(48) NOT NULL,
                shadow_confidence DOUBLE PRECISION NOT NULL,

                -- Soft-PLC Deterministic Action
                plc_action VARCHAR(32) NOT NULL, -- PASS_CONVEYOR, REJECT_PNEUMATIC, REJECT_WATCHDOG
                plc_travel_time_ms DOUBLE PRECISION NOT NULL,

                -- Cloud Async Audit (Store-and-Forward)
                cloud_audit_state VARCHAR(24) NOT NULL DEFAULT 'SKIPPED', -- SKIPPED, QUEUED_STORE_FWD, COMPLETED
                cloud_defect_class VARCHAR(48),
                cloud_confidence DOUBLE PRECISION,
                cloud_latency_ms DOUBLE PRECISION,
                cloud_diagnosis TEXT,

                -- Drift Telemetry
                brightness_mean DOUBLE PRECISION NOT NULL,
                contrast_std DOUBLE PRECISION NOT NULL,
                inspected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS spc_subgroups (
                subgroup_id SERIAL PRIMARY KEY,
                product_recipe VARCHAR(32) NOT NULL,
                batch_code VARCHAR(40) NOT NULL,
                sample_size INTEGER NOT NULL,
                defect_count INTEGER NOT NULL,
                p_value DOUBLE PRECISION NOT NULL,
                p_bar DOUBLE PRECISION NOT NULL,
                ucl DOUBLE PRECISION NOT NULL,
                lcl DOUBLE PRECISION NOT NULL,
                we_rule_violated VARCHAR(64) DEFAULT 'NONE', -- NONE, RULE_1_3SIGMA, RULE_2_2OF3_2SIGMA, RULE_4_8SIDE
                calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS isa18_alarms (
                alarm_id SERIAL PRIMARY KEY,
                alarm_code VARCHAR(48) NOT NULL,
                priority VARCHAR(16) NOT NULL, -- CRITICAL, HIGH, MEDIUM, LOW, DIAGNOSTIC
                state VARCHAR(24) NOT NULL DEFAULT 'ACTIVE_UNACK', -- ACTIVE_UNACK, ACTIVE_ACK, CLEARED
                source_udt VARCHAR(80) NOT NULL,
                message TEXT NOT NULL,
                triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                acked_by VARCHAR(64),
                acked_at TIMESTAMPTZ,
                cleared_at TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS human_feedback (
                feedback_id SERIAL PRIMARY KEY,
                inspection_id BIGINT REFERENCES inspections(inspection_id),
                serial_number VARCHAR(48) NOT NULL,
                product_recipe VARCHAR(32) NOT NULL,
                original_edge_class VARCHAR(48) NOT NULL,
                corrected_status VARCHAR(16) NOT NULL, -- OK, NOK
                corrected_defect_class VARCHAR(48) NOT NULL,
                inspector_user VARCHAR(64) NOT NULL,
                inspector_role VARCHAR(48) NOT NULL,
                notes TEXT,
                used_in_retrain BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id SERIAL PRIMARY KEY,
                actor_user VARCHAR(64) NOT NULL,
                actor_role VARCHAR(48) NOT NULL,
                action_type VARCHAR(64) NOT NULL,
                target_object VARCHAR(120) NOT NULL,
                old_value TEXT,
                new_value TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_inspections_recipe_time ON inspections(product_recipe, inspected_at DESC);
            CREATE INDEX IF NOT EXISTS idx_inspections_batch ON inspections(batch_id);
            CREATE INDEX IF NOT EXISTS idx_inspections_cloud_state ON inspections(cloud_audit_state);
            """
        )

        # Seed Suppliers (idempotent: new recipes add their suppliers to an existing database)
        if True:
            suppliers_seed = [
                ("SUP-MTY-01", "Fundiciones del Norte S.A.", "Monterrey", "Mexico", 25.6866, -100.3161, "CAST_100", 0.04),
                ("SUP-STR-02", "Baden Precision Castings GmbH", "Stuttgart", "Germany", 48.7758, 9.1829, "CAST_100", 0.03),
                ("SUP-DET-03", "Great Lakes Glass & Pharma", "Detroit", "USA", 42.3314, -83.0458, "PACK_200", 0.035),
                ("SUP-OSK-04", "Kansai Sterile Packaging Co.", "Osaka", "Japan", 34.6937, 135.5023, "PACK_200", 0.025),
                ("SUP-SZX-05", "Shenzhen MicroCircuits SMT", "Shenzhen", "China", 22.5431, 114.0579, "PCB_300", 0.045),
                ("SUP-GDL-06", "Jalisco Silicon Valley PCB", "Guadalajara", "Mexico", 20.6597, -103.3496, "PCB_300", 0.038),
                ("SUP-BCN-07", "Iberia Softgel Pharma S.L.", "Barcelona", "Spain", 41.3874, 2.1686, "CAPS_400", 0.03),
                ("SUP-MUM-08", "Deccan Capsule Works", "Mumbai", "India", 19.0760, 72.8777, "CAPS_400", 0.045),
                ("SUP-MTY-09", "Aceros Laminados del Norte", "Monterrey", "Mexico", 25.6866, -100.3161, "STEEL_600", 0.05),
                ("SUP-TSN-10", "Tangshan Hot Strip Mill", "Tangshan", "China", 39.6309, 118.1802, "STEEL_600", 0.06),
                ("SUP-QRO-11", "Bajío Fasteners S.A.", "Querétaro", "Mexico", 20.5888, -100.3899, "SCREW_700", 0.03),
                ("SUP-TPE-12", "Formosa Precision Screw Co.", "Taichung", "Taiwan", 24.1477, 120.6736, "SCREW_700", 0.04),
            ]
            for s in suppliers_seed:
                cur.execute(
                    """
                    INSERT INTO suppliers (supplier_code, name, city, country, lat, lon, specialty_recipe, baseline_defect_rate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (supplier_code) DO NOTHING
                    """,
                    s,
                )

        # Seed one active batch per recipe (idempotent)
        if True:
            batches_seed = [
                ("LOT-CAST-2026-01", "CAST_100", "Impulsor Bomba Centrífuga Hierro Gris", 1, 1000),
                ("LOT-PACK-2026-02", "PACK_200", "Vial Farmacéutico & Cápsula Sellada MVTec", 3, 1200),
                ("LOT-PCB-2026-03", "PCB_300", "Placa Controlador Industrial SMT 4-Capas", 5, 800),
                ("LOT-CAPS-2026-04", "CAPS_400", "Cápsulas de gelatina en blíster (VisA)", 7, 1500),
                ("LOT-STEEL-2026-05", "STEEL_600", "Chapa laminada en caliente Q235 (NEU-DET)", 9, 600),
                ("LOT-SCREW-2026-06", "SCREW_700", "Tornillo Allen M6 en nido (MVTec AD)", 11, 2000),
            ]
            for b in batches_seed:
                cur.execute(
                    """
                    INSERT INTO production_batches (batch_code, product_recipe, product_name, supplier_id, target_qty)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (batch_code) DO NOTHING
                    """,
                    b,
                )

        # Grant strictly SELECT permissions to copilot_ro
        cur.execute(f"GRANT CONNECT ON DATABASE {PG_DB} TO {PG_RO_USER};")
        cur.execute(f"GRANT USAGE ON SCHEMA public TO {PG_RO_USER};")
        cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {PG_RO_USER};")
        cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {PG_RO_USER};")

    conn.close()


if __name__ == "__main__":
    init_database()
    print("PostgreSQL database 'vision_scada' and Read-Only role 'copilot_ro' initialized successfully.")
