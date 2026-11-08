from pathlib import Path
import shutil
import subprocess

html_path = Path(r"C:\dev\VisionSCADA_AI_Twin\static\cell3d_v2.html")
text = html_path.read_text(encoding="utf-8")

old_fetch = "const res = await fetch('/api/state');"
new_fetch = "const apiUrl = (location.port === '8020') ? '/api/state' : 'http://' + (location.hostname || '127.0.0.1') + ':8020/api/state';\n        const res = await fetch(apiUrl);"

if old_fetch in text:
    text = text.replace(old_fetch, new_fetch, 1)
    html_path.write_text(text, encoding="utf-8")
    print("Updated static/cell3d_v2.html")

# Copy to ignition/vision_aoi_cell.html
dest = Path(r"C:\dev\VisionSCADA_AI_Twin\ignition\vision_aoi_cell.html")
dest.write_text(text, encoding="utf-8")
print("Updated ignition/vision_aoi_cell.html")

# Copy into container
res = subprocess.run([
    "docker", "cp",
    str(dest),
    "twin-ignition:/usr/local/bin/ignition/webserver/webapps/main/vision_aoi_cell.html"
], capture_output=True, text=True)
print("docker cp status:", res.returncode, res.stdout, res.stderr)
