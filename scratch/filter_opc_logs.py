import sqlite3

conn = sqlite3.connect(r"C:\dev\VisionSCADA_AI_Twin\scratch\system_logs.idb")
c = conn.cursor()
for row in c.execute("SELECT timestmp, formatted_message, logger_name, level_string FROM logging_event WHERE formatted_message LIKE '%BrewSCADA%' OR formatted_message LIKE '%OPC%' OR logger_name LIKE '%opc%' ORDER BY timestmp DESC LIMIT 50").fetchall():
    print(f"[{row[0]}] {row[3]} [{row[2]}]: {row[1]}")
