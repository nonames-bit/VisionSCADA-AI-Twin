import sqlite3

conn = sqlite3.connect(r"C:\dev\VisionSCADA_AI_Twin\ignition_config.idb")
c = conn.cursor()
tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
opc_tables = [t for t in tables if "OPC" in t.upper() or "SERVER" in t.upper() or "DRIVER" in t.upper()]
print("OPC Tables:", opc_tables)

for t in opc_tables:
    print(f"\n--- Table: {t} ---")
    try:
        col_names = [d[0] for d in c.execute(f"SELECT * FROM {t} LIMIT 1").description]
        print("Columns:", col_names)
        for r in c.execute(f"SELECT * FROM {t}").fetchall():
            print(r)
    except Exception as e:
        print(f"Error reading {t}: {e}")
