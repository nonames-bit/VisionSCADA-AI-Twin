import sqlite3

conn = sqlite3.connect(r"C:\dev\VisionSCADA_AI_Twin\scratch\system_logs.idb")
c = conn.cursor()
tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

for row in c.execute("SELECT timestmp, level, logger, message FROM logging_event ORDER BY timestmp DESC LIMIT 50").fetchall():
    print(f"[{row[0]}] {row[1]} [{row[2]}]: {row[3]}")
