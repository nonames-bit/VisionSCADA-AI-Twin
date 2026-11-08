import sqlite3

conn = sqlite3.connect(r"C:\dev\VisionSCADA_AI_Twin\ignition_config.idb")
c = conn.cursor()
print("OPCSERVERS:")
for r in c.execute("SELECT * FROM OPCSERVERS").fetchall():
    print(r)

print("\nOPCUACONNECTIONSETTINGS:")
for r in c.execute("SELECT * FROM OPCUACONNECTIONSETTINGS").fetchall():
    print(r)

print("\nOPCUACONNECTIONSETTINGS2:")
for r in c.execute("SELECT * FROM OPCUACONNECTIONSETTINGS2").fetchall():
    print(r)
