"""Build the markdown comparison used in the experiment report."""
import json
import sys
from pathlib import Path

rows = {}


def load(name, path, keys):
    data = json.loads(Path(path).read_text())
    for cohort in ("all", "hard", "ordinary", "baseline_over_10m"):
        values = data["metrics"].get(cohort, {})
        for variant in keys:
            if variant in values:
                rows[(cohort, name if variant.startswith("pca") or variant.startswith("gpca")
                      else variant)] = values[variant]


compact_path = sys.argv[1]
pca_path = sys.argv[2]
base_path = sys.argv[3] if len(sys.argv) > 3 else None
tag = sys.argv[4].strip() if len(sys.argv) > 4 else "compact"

table = []
order = ["fp16", "int8_channel", "pca64_int8", "gpca64_int8", tag + "_fp16", tag + "_int8"]
labels = {
    "fp16": "SAM ViT-L FP16 (baseline)",
    "int8_channel": "SAM ViT-L INT8",
    "pca64_int8": "PCA-64 INT8",
    "gpca64_int8": "grad-PCA-64 INT8",
    tag + "_fp16": "compact encoder FP16",
    tag + "_int8": "compact encoder INT8",
}
sources = {}
if base_path:
    sources.update({k: base_path for k in ("fp16", "int8_channel")})
sources.update({"pca64_int8": pca_path, "gpca64_int8": pca_path})
sources.update({tag + "_fp16": compact_path, tag + "_int8": compact_path})

data = {key: json.loads(Path(path).read_text()) for key, path in sources.items()}

for cohort in ("all", "hard", "ordinary", "baseline_over_10m"):
    lines = ["| variant | n | MPE (m) | MOE (deg) | <=1m & <=2deg | >10m |", "|---|---:|---:|---:|---:|---:|"]
    for variant in order:
        block = data.get(variant, {}).get("metrics", {}).get(cohort, {}).get(variant)
        if not block:
            continue
        lines.append("| %s | %d | %.4f | %.4f | %d | %d |" % (
            labels.get(variant, variant), block["n"], block["MPE_m"], block["MOE_deg"],
            block["success_1m_2deg"], block["over_10m"]))
    table.append("### %s\n\n%s\n" % (cohort, "\n".join(lines)))

changes = []
compact = json.loads(Path(compact_path).read_text())["changes"]
for variant in (tag + "_fp16", tag + "_int8"):
    c = compact[variant]
    changes.append("| %s | %+.4f | %+.4f | %d | %d | %d | %.3f | %.2f |" % (
        labels.get(variant, variant), c["delta_MPE_m"], c["delta_MOE_deg"],
        c["success_lost"], c["success_gained"], c["new_over_10m"],
        c["max_extra_position_m"], c["max_extra_rotation_deg"]))
if "pca64_int8" in json.loads(Path(pca_path).read_text()).get("changes", {}):
    pca_changes = json.loads(Path(pca_path).read_text())["changes"]
    for variant in ("pca64_int8", "gpca64_int8"):
        c = pca_changes.get(variant)
        if not c:
            continue
        changes.append("| %s | %+.4f | %+.4f | %d | %d | %d | %.3f | %.2f |" % (
            labels[variant], c["delta_MPE_m"], c["delta_MOE_deg"], c["success_lost"],
            c["success_gained"], c["new_over_10m"], c["max_extra_position_m"],
            c["max_extra_rotation_deg"]))

head = ["| variant | dMPE (m) | dMOE (deg) | success lost | success gained | new >10m | max extra MPE | max extra MOE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
print("\n".join(table))
print("### paired change against the SAM FP16 baseline\n")
print("\n".join(head + changes))
