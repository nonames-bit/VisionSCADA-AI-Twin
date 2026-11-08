import sqlite3

conn = sqlite3.connect(r"C:\dev\VisionSCADA_AI_Twin\ignition_config.idb")
c = conn.cursor()
tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
for t in tables:
    count = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    if count > 0:
        print(f"{t}: {count} rows")
