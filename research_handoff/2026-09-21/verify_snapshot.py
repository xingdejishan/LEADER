import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
errors = []
for item in manifest["files"]:
    path = root / item["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        errors.append(item["path"])
if errors:
    raise SystemExit("Snapshot mismatch: " + ", ".join(errors))
print("Verified " + str(len(manifest["files"])) + " snapshot files")
