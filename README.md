# ChESS — Chemical Engineering Summer Scholars

This repository contains my research for the **Chemical Engineering Summer
Scholars (ChESS)** program. The project investigates relationships between
diffusion and excess entropy using molecular-dynamics simulations in LAMMPS.

The repository currently contains two simulation pipelines:

1. A reduced-unit Lennard-Jones (LJ) reference-fluid pipeline.
2. An all-atom deep eutectic solvent (DES) pipeline for reline, a 1:2 mixture
   of choline chloride and urea.

Both pipelines sweep thermodynamic state points, run LAMMPS, process the
resulting mean-squared displacement (MSD) and radial distribution function
(RDF), and create excess-entropy scaling results.

## Repository layout

### `des_pipeline/`

Automates all-atom simulations of reline using OPLS-DES parameters,
Lennard-Jones plus long-range Coulomb interactions, and PPPM electrostatics.
The current system contains 10 choline cations, 10 chloride anions, and 20 urea
molecules.

- `pipeline.py`: controls temperature and density sweeps, prepares each
  simulation, runs LAMMPS, calculates dimensional and reduced diffusion,
  calculates a pairwise excess-entropy proxy, and creates a Rosenfeld-style
  plot.
- `molecules/`: single-molecule XYZ templates and Packmol's packed structure.
- `forcefield/`: OPLS-DES atom, charge, LJ, bond, angle, and dihedral
  parameters.
- `packmol/`: template used to pack molecules into a density-dependent box.
- `scripts/build_data.py`: converts packed coordinates and force-field
  parameters into a LAMMPS `data.reline` file.
- `templates/in.des.template`: LAMMPS input template for minimization, NVT
  equilibration, production, RDF, and MSD.
- `NOTES.md`: detailed explanation of the complete DES workflow.
- `templates/NOTES.md`: line-by-line explanation of the LAMMPS input.
- `forcefield/NOTES.md`: force-field sources and atom-type descriptions.
- `output/`: timestamped simulation results created locally and ignored by
  Git.


### `lj_pipeline/`

Automates simulations of a Lennard-Jones reference fluid in reduced LJ units.

- `pipeline.py`: creates state-point inputs, runs LAMMPS, analyzes results, and
  plots diffusion against pairwise excess entropy.
- `templates/in.lj.template`: LAMMPS input template.
- `output/`: timestamped simulation results created locally and ignored by
  Git.

## Requirements

The pipelines use:

- Python 3;
- NumPy;
- Matplotlib;
- SciPy;
- LAMMPS, available through the `lmp` command; and
- MPI, available through `mpirun`.

The DES pipeline additionally requires Packmol through the `packmol` command.

## Running the DES pipeline

```bash
cd des_pipeline
python3 pipeline.py
```

## Running the LJ pipeline

```bash
cd lj_pipeline
python3 pipeline.py
```

Each launch creates a timestamped folder under that pipeline's `output/`
directory. Large simulation outputs are intentionally excluded from version
control.

## Acknowledgments

Special thanks to Professor Jerry (Gerald) Wang in Carnegie Mellon
University's Department of Civil and Environmental Engineering for his
guidance throughout this research, as well as to the Chemical Engineering 
department at Carnegie Mellon University for their gracious funding.