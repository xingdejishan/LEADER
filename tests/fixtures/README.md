# Frozen server references

Generated on the unchanged server with PyTorch 1.12.0+cu116, Python 3.8.20 and RTX3090. Reference indices are outputs of the original CUDA `torch.argsort`, not of `server_argsort_numpy`.

`server_torch112_sort.npz`: NumPy RNG seed 481516; lengths 0,1,2,20,30,32,33,40,64,65,128,129,256,352,499,1024,1025,2048,2049,3000. Four rows each: all equal, increasing, two integer-valued random rows; ascending and descending reference indices.

`server_sc2_sort_trace.npz`: frames 0,100,141,204,263,438,500,641,825,851,867,904 from the fixed 905-frame cache; descending and random_2089; three sorting stages per case. The eight high-disagreement frames are diagnostic examples, not an unbiased accuracy sample.

Full paired-input validation results live in the workspace at `research/server_sort_compat/`.
