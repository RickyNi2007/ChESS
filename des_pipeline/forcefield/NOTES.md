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