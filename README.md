# ChESS

Research project on excess-entropy scaling for Lennard-Jones fluids (DES work planned later). Special thanks to Professor Jerry (Gerald) Wang at CMU's Civil and Environmental Engineering for leading me through this entire process. Thanks to Claude and Cursor for helping me code and uploading this to GitHub.

## Layout

- `lj_pipeline/` — LJ automation (`pipeline.py` + LAMMPS templates)
- `des_pipeline/` — future DES pipeline (placeholder)
- `docs/` — research notes
- `results/` — local outputs (not tracked by Git)

## Run LJ pipeline

```bash
cd lj_pipeline
python3 pipeline.py