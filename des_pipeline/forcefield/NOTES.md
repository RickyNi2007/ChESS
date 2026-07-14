## urea
- Geometry: DES/PDBs/urea.pdb
- Topology: DES/ITP/urea_DES.itp
- LJ types: DES/ITP/urea_atomtypes_DES.itp
- Net charge checked: 0 (verify via summation of atom charges)

## chloride
- Geometry: DES/PDBs/chloride.pdb
- Topology: DES/ITP/cl_DES.itp
- LJ types: DES/ITP/cl_atomtypes_DES.itp
- Net charge checked: -0.800 (OPLS-DES scaled ion charge, not -1.0)

## choline
- Geometry: DES/PDBs/choline.pdb
- Topology: DES/ITP/choline_DES.itp
- LJ types: DES/ITP/choline_atomtypes_DES.itp
- Net charge checked: +0.800 (cancels chloride -0.800)
- Note: most H types have nonzero LJ here; only HY (hydroxyl H) has ε=σ=0

## packmol
- Input: packmol/pack_reline.inp
- N = 10 choline, 10 chloride, 20 urea (1:2 ChCl:urea)
- Box: 24 Å cube (24,24,24), tolerance 2.0 Å
- Output: molecules/reline_pack.xyz (380 atoms)

## OPLS-DES atom types (names → meaning)

LAMMPS integer types are assigned in build_data.py in this order
(first time each name appears). Approximate LAMMPS IDs:

### 1. Choline (types 1–9)
| LAMMPS id | name | element | meaning |
|-----------|------|---------|---------|
| 1 | CS | C | CH2 carbon next to N (toward ethyl alcohol arm) |
| 2 | HS | H | hydrogens on CS |
| 3 | NA | N | quaternary ammonium nitrogen (+ charged center) |
| 4 | CW | C | CH2 carbon next to OH |
| 5 | HW | H | hydrogens on CW |
| 6 | OY | O | hydroxyl oxygen |
| 7 | HY | H | hydroxyl hydrogen (ε=σ=0) |
| 8 | CA | C | methyl carbons on N |
| 9 | HA | H | methyl hydrogens |

### 2. Chloride (type 10)
| LAMMPS id | name | element | meaning |
|-----------|------|---------|---------|
| 10 | Cl | Cl | chloride anion (scaled charge −0.8) |

### 3. Urea (types 11–15)
| LAMMPS id | name | element | meaning |
|-----------|------|---------|---------|
| 11 | C | C | carbonyl carbon |
| 12 | O | O | carbonyl oxygen |
| 13 | N | N | amine nitrogens |
| 14 | HT | H | amine H (trans-like in Acevedo naming); ε=σ=0 |
| 15 | HC | H | amine H (cis-like in Acevedo naming); ε=σ=0 |

Notes:
- `type` is the FF class; `elem` is the chemical element for XYZ/display.
- Urea `C`/`N`/`O` are different types from choline carbons/nitrogen even when the element matches.
- Confirm IDs by printing `type_of` from des_pipeline/scripts/build_data.py