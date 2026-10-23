#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe 5: the full-run ledger's s/d (the per-frame branch's own offsets) - regime check."""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']

S, D, G, EXP, RUN = [], [], [], [], []
n_rows = 0
for name in FILES:
    with open(P + name) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get('key') or not r.get('seconds'):
                continue
            n_rows += 1
            S.append(r.get('s') if r.get('s') is not None else np.nan)
            D.append(r.get('d') if r.get('d') is not None else np.nan)
            G.append(r.get('gain') if r.get('gain') is not None else np.nan)
            EXP.append(r.get('expanded') if r.get('expanded') is not None else -1)
            RUN.append(bool(r.get('ran_fine')))
S = np.array(S, float); D = np.array(D, float); G = np.array(G, float)
EXP = np.array(EXP, float); RUN = np.array(RUN, bool)
print('[rows] %d   ran_fine %d (%.1f%%)  gate2 field sample %s'
      % (n_rows, RUN.sum(), 100 * RUN.mean(), FILES))

print('\n[gain] median %.4f  p90 %.4f  p99 %.4f  >0 %.1f%%  >0.01 %.1f%%'
      % (np.nanmedian(G), np.nanpercentile(G, 90), np.nanpercentile(G, 99),
         100 * np.nanmean(G > 0), 100 * np.nanmean(G > 0.01)))
sel = np.isfinite(S)
print('\n[s] n=%d  |s|<1e-9 %.1f%%  |s|>0: median %.2f p99 %.2f  hist %s'
      % (sel.sum(), 100 * np.mean(np.abs(S[sel]) < 1e-9),
         np.median(np.abs(S[sel][S[sel] != 0])) if (S[sel] != 0).any() else 0,
         np.percentile(np.abs(S[sel]), 99),
         np.histogram(S[sel], bins=[-8, -6, -4, -2, -0.5, 0.5, 2, 4, 6, 8])[0].tolist()))
print('[d] n=%d  |d|<1e-9 %.1f%%  hist %s'
      % (sel.sum(), 100 * np.mean(np.abs(D[sel]) < 1e-9),
         np.histogram(D[sel], bins=[-2, -1.2, -0.6, -0.15, 0.15, 0.6, 1.2, 2])[0].tolist()))
print('[expanded] median %.0f  mean %.2f' % (np.median(EXP[EXP >= 0]), EXP[EXP >= 0].mean()))

big = sel & (np.abs(S) >= 1.0)
print('\n|s|>=1 m: %d frames (%.1f%%)   their gain median %.4f' % (
    big.sum(), 100 * big.sum() / sel.sum(), np.nanmedian(G[big])))
print('gain quartiles on |s|>=1: %s' % (np.round(np.nanpercentile(G[big], [10, 50, 90]), 4),))
