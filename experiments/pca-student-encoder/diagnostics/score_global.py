"""Score a descriptor matrix with the same recall protocol as the DINOv2 benchmark."""
import sys

import numpy as np

BENCH = "/home/zhangcy/projects/nclt-cand-audit-20261023-v1/src/benchmark.npz"
GLOBAL = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/zhangcy/projects/nclt-cand-audit-20261023-v1/results/global_descriptors.npz"
RADII = (5.0, 10.0, 25.0, 50.0)
TOPK = (1, 5, 10, 50)

bench = np.load(BENCH)
query_positions = bench["query_positions"]
reference_positions = bench["reference_positions"]
bench_q = [str(k) for k in bench["query_keys"]]
bench_r = [str(k) for k in bench["reference_keys"]]

data = np.load(GLOBAL)
query_keys = [str(k) for k in data["query_keys"]]
reference_keys = [str(k) for k in data["reference_keys"]]
index_r = {k: i for i, k in enumerate(reference_keys)}
index_q = {k: i for i, k in enumerate(query_keys)}
missing_r = [k for k in bench_r if k not in index_r]
missing_q = [k for k in bench_q if k not in index_q]
print("reference coverage %d/%d, query coverage %d/%d" % (
    len(bench_r) - len(missing_r), len(bench_r), len(bench_q) - len(missing_q), len(bench_q)))
keep_q = [i for i, k in enumerate(bench_q) if k in index_q]

for name in ("mean", "max"):
    ref = data["reference_" + name][[index_r[k] for k in bench_r]]
    qry = data["query_" + name][[index_q[bench_q[i]] for i in keep_q]]
    sim = qry @ ref.T
    order = np.argsort(-sim, axis=1)[:, :max(TOPK)]
    distance = np.linalg.norm(query_positions[keep_q][:, None, :] - reference_positions[order], axis=-1)
    print("== compact %s pooling (n=%d) ==" % (name, len(keep_q)))
    for radius in RADII:
        cells = []
        for k in TOPK:
            cells.append("top%-3d %.3f" % (k, float((distance[:, :k].min(axis=1) <= radius).mean())))
        print("  radius %4.0fm  " % radius + "  ".join(cells))
