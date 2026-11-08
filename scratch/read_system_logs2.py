import sqlite3

conn = sqlite3.connect(r"C:\dev\VisionSCADA_AI_Twin\scratch\system_logs.idb")
c = conn.cursor()
col_names = [d[0] for d in c.execute("SELECT * FROM logging_event LIMIT 1").description]
print("Columns:", col_names)
for row in c.execute("SELECT * FROM logging_event ORDER BY timestmp DESC LIMIT 30").fetchall():
    print(row)
