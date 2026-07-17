# Reline DES Pipeline

This document explains how `pipeline.py` connects the molecular files, OPLS-DES
force-field parameters, Packmol, `build_data.py`, LAMMPS, and the final
Rosenfeld-style analysis.

The simulated liquid is **reline**, a 1:2 mixture of choline chloride and urea.
The current box contains 10 choline cations, 10 chloride anions, and 20 urea
molecules (380 atoms total).

## Overall workflow

For every requested temperature and mass density, the pipeline:

1. Calculates the required cubic box length.
2. Uses Packmol to place the molecules in that box.
3. Builds a complete LAMMPS `data.reline` file.
4. Creates a state-point folder and fills the LAMMPS input template.
5. Runs energy minimization, NVT equilibration, and NVT production.
6. Reads the mean-squared displacement (MSD) and radial distribution function
   (RDF).
7. Calculates the diffusion coefficient, pairwise excess entropy, and reduced
   diffusion coefficient.
8. Writes a run summary and Rosenfeld-style plot.

## Step 0 — User-controlled constants

The main simulation controls are near the top of `pipeline.py`:

- `temp_start`, `temp_end`, `temp_step`: temperature sweep in K.
- `density_start`, `density_end`, `density_step`: mass-density sweep in
  g/cm³.
- `eq_steps`: number of NVT equilibration timesteps.
- `prod_steps`: number of NVT production timesteps.
- `total_mass_amu`: total mass of the 10-choline, 10-chloride, 20-urea box.

The temperature and density arrays are generated with `numpy.arange()`. Its end
value is intended to be exclusive, although decimal floating-point values can
occasionally make a boundary value appear. Check the printed state points when
changing a sweep.

`total_mass_amu` must be updated if the number or identity of the molecules is
changed. It is required to convert a requested mass density into a box volume.

## Step 1 — Start a timestamped pipeline run

Running:

```bash
python pipeline.py
```

calls `main()`. It creates one folder for the complete sweep:

```text
output/run_YYYYMMDD_HHMMSS/
```

It then loops over every `(temperature, density)` combination. Each combination
is called a **state point** and receives its own subfolder, for example:

```text
output/run_YYYYMMDD_HHMMSS/T298.0_rho1.200/
```

`summary.dat` and `plot.png` are written to the timestamped run folder after
all state points finish successfully.

## Step 2 — Convert density into box length

`box_length_angstrom()` calculates the side length `L` of a cubic box. The
number of molecules and their total mass remain fixed, so:

```text
mass = total_mass_amu × grams_per_amu
volume = mass / mass_density
L = cube_root(volume)
```

The result is converted from cm to Å. A higher density gives a smaller box,
while a lower density gives a larger box.

The pipeline uses NVT dynamics, so the volume is fixed during the simulation.
This means density is an input established through `L`, not a quantity adjusted
by an NPT barostat.

## Step 3 — Pack molecules with Packmol

`run_packmol(L)`:

1. Reads `packmol/pack_reline.inp.template`.
2. Replaces `BOX_PLACEHOLDER` with the calculated box length.
3. Writes the generated input to `packmol/pack_reline.inp`.
4. Runs Packmol and records its terminal output in `packmol/packmol.log`.

The Packmol input requests:

- 10 copies of `molecules/choline.xyz`;
- 10 copies of `molecules/chloride.xyz`;
- 20 copies of `molecules/urea.xyz`; and
- a 2.0 Å minimum inter-molecular atom separation.

Packmol writes the packed coordinates to `molecules/reline_pack.xyz`. Packmol
only creates coordinates: it does not assign charges, force-field types, bonds,
angles, or dihedrals.

To permanently change the packing tolerance, edit
`packmol/pack_reline.inp.template`. The generated `pack_reline.inp` is
overwritten at every state point.

## Step 4 — Build the LAMMPS data file

`run_build_data(L)` runs:

```bash
python scripts/build_data.py L
```

`build_data.py` reads:

- `molecules/reline_pack.xyz`: the coordinates produced by Packmol;
- `forcefield/choline.params`;
- `forcefield/chloride.params`; and
- `forcefield/urea.params`.

The parameter files contain:

- **Atoms:** force-field type name, element, mass, partial charge, LJ epsilon,
  and LJ sigma.
- **Bonds:** bonded local atom IDs, equilibrium distance `r0`, and force
  constant `K`.
- **Angles:** local atom IDs `i-j-k`, equilibrium angle `theta0`, and `K`.
- **Dihedrals:** local atom IDs `i-j-k-l` and OPLS coefficients `V1..V4`.

The script builds these Python data structures in memory:

- an atom-type map that translates force-field names such as `CS` and `Cl`
  into integer LAMMPS atom types;
- atom records containing atom ID, molecule ID, integer type, charge, and
  coordinates;
- box-wide bond, angle, and dihedral topology, with each molecule's local atom
  IDs shifted to global IDs; and
- unique bonded parameter-type tables.

It writes `data/data.reline` in the plain-text LAMMPS data-file format for
`atom_style full`. This file contains box boundaries, masses, atoms, charges,
coordinates, molecular topology, and bonded coefficients.

`data/data.reline` is a staging file and is overwritten for each state point.
The pipeline preserves the correct version by copying it into the state-point
output folder before moving to the next point.

## Step 5 — Prepare one state-point folder

`prepare_run_dir()`:

1. Calculates `L`.
2. Runs Packmol.
3. Runs `build_data.py`.
4. Creates the state-point folder.
5. Copies `data/data.reline` into it as `data.reline`.
6. Reads `templates/in.des.template` and replaces:
   - `TEMP_PLACEHOLDER`;
   - `EQ_STEPS_PLACEHOLDER`; and
   - `PROD_STEPS_PLACEHOLDER`.
7. Writes the filled input as `in.des`.

The state-point folder is therefore a self-contained LAMMPS job containing the
correct box and settings for that temperature and density.

## Step 6 — Run LAMMPS

`run_lammps()` runs the following command from inside the state-point folder:

```bash
mpirun -np 4 lmp -in in.des
```

The four MPI processes share the LAMMPS calculation. The generated files
include:

- `experiment.log`: terminal output captured by `pipeline.py`;
- `log.lammps`: LAMMPS's standard log;
- `msd.dat`: time and mean-squared displacement;
- `rdf.out`: time-averaged radial distribution function;
- `in.des`: the filled LAMMPS script; and
- `data.reline`: the initial system and topology.

See `templates/NOTES.md` for a detailed explanation of the LAMMPS commands.

## Step 7 — Calculate dimensional diffusion

`parse_msd()` reads the two columns in `msd.dat`: time in fs and total MSD in
Å².

`compute_diffusion()` fits a straight line to the final 60% of the MSD data.
For three-dimensional diffusion, the Einstein relation is:

```text
MSD = 6 D t
D = fitted_MSD_slope / 6
```

The resulting dimensional diffusion coefficient has units of Å²/fs. A fitted
slope is used instead of dividing one MSD value by one time value because the
fit is less sensitive to individual noisy samples.

## Step 8 — Calculate pairwise excess entropy

`parse_rdf()` reads `rdf.out` and retains the last time-averaged RDF block.
`compute_s2()` combines this `g(r)` with the atomic number density:

```text
s2/kB = -2 pi rho_n integral[(g ln(g) - g + 1) r^2 dr]
```

This is a pair-correlation contribution (or proxy) based on the combined
all-atom RDF. It is not the complete thermodynamic excess entropy of the
mixture.

## Step 9 — Calculate reduced diffusion

`reduce_diffusion()` converts the dimensional diffusion coefficient to
Rosenfeld-style reduced diffusion:

```text
D* = D rho_n^(1/3) / sqrt(kB T / m)
```

The function first converts Å²/fs to m²/s, number density to 1/m³, and average
atomic mass to kg. `D*` is dimensionless, allowing points at different
temperatures and densities to be compared on the same scaling plot.

## Step 10 — Write the summary and plot

After every state point finishes, `main()` writes:

```text
output/run_YYYYMMDD_HHMMSS/summary.dat
```

Its columns are:

- temperature in K;
- mass density in g/cm³;
- dimensional diffusion in Å²/fs;
- reduced diffusion `D*`; and
- pairwise excess entropy `s2/kB`.

`plot_rosenfeld()` writes:

```text
output/run_YYYYMMDD_HHMMSS/plot.png
```

It plots `D*` against `s2/kB` on a logarithmic y-axis. A regression is performed
on `ln(D*)` versus `s2/kB`, and the resulting exponential best-fit curve,
equation, R² value, temperature range, and density range are displayed.



