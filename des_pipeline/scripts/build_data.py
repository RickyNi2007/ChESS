"""Builds LAMMPS data.reline from Packmol XYZ + OPLS-DES params"""

from pathlib import Path
import sys

# Paths relative to des_pipeline/ when you run:
ROOT = Path(__file__).resolve().parent.parent
#Clarification Comments:
    #ROOT = path(path_of_file).absolute_path.one_folder_above.one_folder_above
    #ROOT = des_pipeline (absolute path, works no matter what file your terminal is current located)
MOLECULES = ROOT / "molecules"
FORCEFIELD = ROOT / "forcefield"
OUT = ROOT / "data" / "data.reline"
PACKED_XYZ = MOLECULES / "reline_pack.xyz"

#Constants
N_CHOLINE = 10
N_CHLORIDE = 10
N_UREA = 20
ATOMS_PER_CHOLINE = 21
ATOMS_PER_CHLORIDE = 1
ATOMS_PER_UREA = 8
BOX_LO = 0.0
BOX_HI = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0 #e.g python scripts/build_data.py 15.32, 15.32 is manually typed and passed in a terminal command

def read_xyz(path: Path):
    """Return (comment, list of (element, x, y, z))."""
    lines = path.read_text().strip().splitlines() #read_text() is a Path method
    n = int(lines[0])
    comment = lines[1]
    atoms = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        element = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((element, x, y, z))
    if len(atoms) != n:
        raise ValueError(f"Expected {n} atoms, got {len(atoms)}")
    return comment, atoms 

def parse_params(path: Path):
    """
    Parse our forcefield *.params file.
    Returns dict with keys: atoms, bonds, angles, dihedrals
    Each is a list of dicts.
    """
    section = None
    data = {"atoms": [], "bonds": [], "angles": [], "dihedrals": []}

    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip() #reads every line, if it has # or [] continues 
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower() #e.g [atoms] == atoms
            continue
        if section is None:
            continue
        #only lines with data (only numbers) survives these if statements

        parts = line.split()
        if section == "atoms":
            # id type elem mass charge epsilon sigma
            data["atoms"].append({
                "id": int(parts[0]),
                "type": parts[1],
                "elem": parts[2],
                "mass": float(parts[3]),
                "charge": float(parts[4]),
                "epsilon": float(parts[5]),
                "sigma": float(parts[6]),
            })
        elif section == "bonds":
            # id i j r0 K
            data["bonds"].append({
                "id": int(parts[0]),
                "i": int(parts[1]),
                "j": int(parts[2]),
                "r0": float(parts[3]),
                "K": float(parts[4]),
            })
        elif section == "angles":
            # id i j k theta0 K
            data["angles"].append({
                "id": int(parts[0]),
                "i": int(parts[1]),
                "j": int(parts[2]),
                "k": int(parts[3]),
                "theta0": float(parts[4]),
                "K": float(parts[5]),
            })
        elif section == "dihedrals":
            # id i j k l V1 V2 V3 V4  (ignore note words after)
            data["dihedrals"].append({
                "id": int(parts[0]),
                "i": int(parts[1]),
                "j": int(parts[2]),
                "k": int(parts[3]),
                "l": int(parts[4]),
                "V1": float(parts[5]),
                "V2": float(parts[6]),
                "V3": float(parts[7]),
                "V4": float(parts[8]),
            })
    return data #a dictionary of lists with dictionaries in them

def build_type_maps(choline, chloride, urea):
    """
    Assign unique LAMMPS atom types in order: choline, chloride, urea.
    Returns:
      type_of: dict name -> type_id (int)
      mass_of: dict type_id -> mass
      eps_sig: dict type_id -> (epsilon, sigma)
    """
    type_of = {}
    mass_of = {}
    eps_sig = {}
    next_type = 1

    def add_species(params):
        nonlocal next_type #changes the next_type variable in build_type_maps to be nonlocal variable
        for atom in params["atoms"]:
            name = atom["type"]
            if name in type_of:
                continue  # already have this type name
            tid = next_type
            next_type += 1
            type_of[name] = tid
            mass_of[tid] = atom["mass"]
            eps_sig[tid] = (atom["epsilon"], atom["sigma"])

    add_species(choline)
    add_species(chloride)
    add_species(urea)
    return type_of, mass_of, eps_sig

def build_atom_records(atoms, choline, chloride, urea, type_of):
    """
    Flatten packed XYZ into LAMMPS atom rows.
    Returns list of dicts: id, mol, type, q, x, y, z
    """
    records = []
    atom_id = 1
    mol_id = 1
    idx = 0  # index into packed atoms list

    # cholines, N_CHLORINE molecules and 21 atoms per molecule
    for _ in range(N_CHOLINE):
        for local in choline["atoms"]:
            elem, x, y, z = atoms[idx]
            idx += 1
            records.append({
                "id": atom_id,
                "mol": mol_id,
                "type": type_of[local["type"]],
                "q": local["charge"],
                "x": x, "y": y, "z": z,
            })
            atom_id += 1
        mol_id += 1

    # chlorides
    for _ in range(N_CHLORIDE):
        local = chloride["atoms"][0]
        elem, x, y, z = atoms[idx]
        idx += 1
        records.append({
            "id": atom_id,
            "mol": mol_id,
            "type": type_of[local["type"]],
            "q": local["charge"],
            "x": x, "y": y, "z": z,
        })
        atom_id += 1
        mol_id += 1

    # ureas, N_UREA ureas and 8 atoms per urea
    for _ in range(N_UREA):
        for local in urea["atoms"]:
            elem, x, y, z = atoms[idx]
            idx += 1
            records.append({
                "id": atom_id,
                "mol": mol_id,
                "type": type_of[local["type"]],
                "q": local["charge"],
                "x": x, "y": y, "z": z,
            })
            atom_id += 1
        mol_id += 1

    if idx != len(atoms):
        raise ValueError("Atom index mismatch while building records")
    return records


def write_full_data(
    path, records, mass_of,
    bonds, angles, dihedrals,
    bond_type_of, angle_type_of, dihedral_type_of,
):
    """
    Write a complete LAMMPS data file:
    header, box, Masses, Atoms, Bonds, Angles, Dihedrals, and bonded coeffs.
    """
    n_atoms = len(records)
    n_atom_types = len(mass_of)
    n_bonds = len(bonds)
    n_angles = len(angles)
    n_dihedrals = len(dihedrals)
    n_bond_types = len(bond_type_of)
    n_angle_types = len(angle_type_of)
    n_dihedral_types = len(dihedral_type_of)

    lines = []
    lines.append("LAMMPS data file for reline (OPLS-DES), built by build_data.py\n")
    lines.append("\n")
    lines.append(f"{n_atoms} atoms\n")
    lines.append(f"{n_bonds} bonds\n")
    lines.append(f"{n_angles} angles\n")
    lines.append(f"{n_dihedrals} dihedrals\n")
    lines.append("\n")
    lines.append(f"{n_atom_types} atom types\n")
    lines.append(f"{n_bond_types} bond types\n")
    lines.append(f"{n_angle_types} angle types\n")
    lines.append(f"{n_dihedral_types} dihedral types\n")
    lines.append("\n")
    lines.append(f"{BOX_LO:.6f} {BOX_HI:.6f} xlo xhi\n")
    lines.append(f"{BOX_LO:.6f} {BOX_HI:.6f} ylo yhi\n")
    lines.append(f"{BOX_LO:.6f} {BOX_HI:.6f} zlo zhi\n")

    lines.append("\nMasses\n\n")
    for tid in sorted(mass_of):
        lines.append(f"{tid} {mass_of[tid]:.4f}\n")

    lines.append("\nAtoms  # full\n\n")
    for r in records:
        lines.append(
            f"{r['id']} {r['mol']} {r['type']} {r['q']:.6f} "
            f"{r['x']:.6f} {r['y']:.6f} {r['z']:.6f}\n"
        )

    lines.append("\nBonds\n\n")
    for bid, b in enumerate(bonds, start=1):
        lines.append(f"{bid} {b['type']} {b['i']} {b['j']}\n")

    lines.append("\nAngles\n\n")
    for aid, a in enumerate(angles, start=1):
        lines.append(f"{aid} {a['type']} {a['i']} {a['j']} {a['k']}\n")

    lines.append("\nDihedrals\n\n")
    for did, d in enumerate(dihedrals, start=1):
        lines.append(
            f"{did} {d['type']} {d['i']} {d['j']} {d['k']} {d['l']}\n"
        )

    # Invert type maps: type_id -> parameter key
    bond_params = {tid: key for key, tid in bond_type_of.items()}
    angle_params = {tid: key for key, tid in angle_type_of.items()}
    dihedral_params = {tid: key for key, tid in dihedral_type_of.items()}

    lines.append("\nBond Coeffs\n\n")
    for tid in sorted(bond_params):
        r0, K = bond_params[tid]
        lines.append(f"{tid} {K:.4f} {r0:.4f}\n")  # LAMMPS harmonic: K r0

    lines.append("\nAngle Coeffs\n\n")
    for tid in sorted(angle_params):
        theta0, K = angle_params[tid]
        lines.append(f"{tid} {K:.4f} {theta0:.4f}\n")  # LAMMPS harmonic: K theta0

    lines.append("\nDihedral Coeffs\n\n")
    for tid in sorted(dihedral_params):
        V1, V2, V3, V4 = dihedral_params[tid]
        lines.append(f"{tid} {V1:.4f} {V2:.4f} {V3:.4f} {V4:.4f}\n")  # opls

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))

def get_or_add_type(table, key, next_id_holder):
    """
    table: dict key -> type_id; next_id_holder: single-item list [n].
    If these parameters have been used before, reuse the integer.
    If not, create a new integer and remember it in the table.
    Used in append_mol_topology
    """
    if key not in table:
        table[key] = next_id_holder[0]
        next_id_holder[0] += 1
    return table[key]


def append_mol_topology(
    bonds, angles, dihedrals,
    template, first_atom_id,
    bond_type_of, angle_type_of, dihedral_type_of,
    next_bond_type, next_angle_type, next_dihedral_type,
):
    """
    Add bonded terms for one molecule copy. template = choline/urea params.
    Go through params bonds, angles, and dihedrals. 
    Append them to empty lists and return them.
    Used in build_topology
    """
    for b in template["bonds"]:
        t = get_or_add_type(bond_type_of, (b["r0"], b["K"]), next_bond_type)
        bonds.append({
            "type": t,
            "i": first_atom_id + b["i"] - 1,
            "j": first_atom_id + b["j"] - 1,
            "r0": b["r0"],
            "K": b["K"],
        })
    for a in template["angles"]:
        t = get_or_add_type(angle_type_of, (a["theta0"], a["K"]), next_angle_type)
        angles.append({
            "type": t,
            "i": first_atom_id + a["i"] - 1,
            "j": first_atom_id + a["j"] - 1,
            "k": first_atom_id + a["k"] - 1,
            "theta0": a["theta0"],
            "K": a["K"],
        })
    for d in template["dihedrals"]:
        key = (d["V1"], d["V2"], d["V3"], d["V4"])
        t = get_or_add_type(dihedral_type_of, key, next_dihedral_type)
        dihedrals.append({
            "type": t,
            "i": first_atom_id + d["i"] - 1,
            "j": first_atom_id + d["j"] - 1,
            "k": first_atom_id + d["k"] - 1,
            "l": first_atom_id + d["l"] - 1,
            "V1": d["V1"], "V2": d["V2"], "V3": d["V3"], "V4": d["V4"],
        })

def build_topology(choline, chloride, urea):
    """
    Builds the topology of the box's molecules
    """
    bonds, angles, dihedrals = [], [], []
    bond_type_of, angle_type_of, dihedral_type_of = {}, {}, {}
    next_bt, next_at, next_dt = [1], [1], [1]

    atom_id = 1
    # cholines
    for _ in range(N_CHOLINE):
        append_mol_topology(
            bonds, angles, dihedrals, choline, atom_id,
            bond_type_of, angle_type_of, dihedral_type_of,
            next_bt, next_at, next_dt,
        )
        atom_id += ATOMS_PER_CHOLINE
    # chlorides: no bonds
    atom_id += N_CHLORIDE * ATOMS_PER_CHLORIDE
    # ureas
    for _ in range(N_UREA):
        append_mol_topology(
            bonds, angles, dihedrals, urea, atom_id,
            bond_type_of, angle_type_of, dihedral_type_of,
            next_bt, next_at, next_dt,
        )
        atom_id += ATOMS_PER_UREA

    return bonds, angles, dihedrals, bond_type_of, angle_type_of, dihedral_type_of

def main():
    print("ROOT =", ROOT)
    print("Will write:", OUT)

    comment, atoms = read_xyz(PACKED_XYZ) # atoms is a tuple of len == 4, with xyz coords 
    print("XYZ comment:", comment)
    print("Atom count:", len(atoms))
    print("First atom:", atoms[0])
    print("Last atom:", atoms[-1])

    expected = (N_CHOLINE * ATOMS_PER_CHOLINE + N_CHLORIDE * ATOMS_PER_CHLORIDE + N_UREA * ATOMS_PER_UREA) #expected # of atoms
    if len(atoms) != expected:
        raise ValueError(f"Expected {expected} atoms, got {len(atoms)}")
    first_cl = N_CHOLINE * ATOMS_PER_CHOLINE  # 210
    first_urea = first_cl + N_CHLORIDE * ATOMS_PER_CHLORIDE  # 220
    print("First chloride atom:", atoms[first_cl])
    print("First urea atom:", atoms[first_urea])

    choline = parse_params(FORCEFIELD / "choline.params")
    chloride = parse_params(FORCEFIELD / "chloride.params")
    urea = parse_params(FORCEFIELD / "urea.params")
    print("Choline atoms/bonds:", len(choline["atoms"]), len(choline["bonds"]))
    print("Chloride atoms/bonds:", len(chloride["atoms"]), len(chloride["bonds"]))
    print("Urea atoms/bonds:", len(urea["atoms"]), len(urea["bonds"]))
    print("Choline net charge:", sum(a["charge"] for a in choline["atoms"]))
    print("Chloride charge:", chloride["atoms"][0]["charge"])
    print("Urea net charge:", sum(a["charge"] for a in urea["atoms"]))

    type_of, mass_of, eps_sig = build_type_maps(choline, chloride, urea)
    print("Type map:", type_of)
    print("Number of atom types:", len(type_of))

    records = build_atom_records(atoms, choline, chloride, urea, type_of)
    bonds, angles, dihedrals, bto, ato, dto = build_topology(choline, chloride, urea)
    print("Bonds/angles/dihedrals:", len(bonds), len(angles), len(dihedrals))
    
    write_full_data(OUT, records, mass_of, bonds, angles, dihedrals, bto, ato, dto)
    print("Wrote", OUT)
    print("Total charge:", sum(r["q"] for r in records))
    print("First record:", records[0])
    print("First Cl record:", records[210])


if __name__ == "__main__":
    main()