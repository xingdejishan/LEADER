# Existing NCLT split: raw scan retrieval

The existing 907/303/148 image split is recorded in `/home/zhang/rscore-l-local/data/manifest.json` under WSL. The raw scan source on the user-provided SSH server is `/root/rivermind-data/datasets/NCLT/<date>/velodyne_sync/`.

`fetch_training_scans.py` downloads only exact-timestamp training scans using four workers and short compressed SSH batches, without making a server-side archive. Verified files survive connection failures and are skipped on resume. It can also recover complete, hash-verified members from the initial interrupted archive transfers. The password is prompted or passed in memory and is never written to the source or reports. SHA256 is computed on the server's original bytes and checked again against each locally written file. Existing conflicting scans cause an error rather than being overwritten. Archive member names must match the requested split, and paths are checked against the destination directory before writing.

The local target is `glace-local/data/scans/<date>/velodyne_sync/` in the shared workspace. Per-part compressed archives and the combined audit are stored outside this Git repository in `glace-local/data/train907_raw_scan_audit/`.

```powershell
python LEADER/research/nclt_data/fetch_training_scans.py
python LEADER/research/nclt_data/verify_pairs.py
```

`verify_pairs.py` checks the image SHA256 against the frozen image split, the shape and finiteness of camera poses/intrinsics, raw scan byte alignment, and training scan hashes against the server. It writes `paired_manifest.json` and `pairing_summary.json`. Every original image row is preserved; missing raw scans are explicitly marked rather than being replaced by nearby timestamps. This retrieval does not train a model or alter the original split.

Server inventory found 319/320 requested scans for 2012-01-22, 273/273 for 2012-02-02, and 313/314 for 2012-05-11. The exact files `1327251814000992.bin` and `1336763674005996.bin` were not found in the source directories or the other searched server experiment directories. Their nearest source scans differ by approximately 0.2 seconds and were not substituted. The original downloaded archives were not retained on the server, so this audit does not establish whether those files are absent from the official upstream archives.
