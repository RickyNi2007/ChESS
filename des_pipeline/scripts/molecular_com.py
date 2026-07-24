"""
Molecular center-of-mass (COM) analysis for Reline DES trajectories.

WHY: heavy-site proxies (choline N, Cl, urea C) approximate molecular motion.
True COM uses every atom in a molecule, mass-weighted:

    R_com = (sum_i m_i * r_i) / (sum_i m_i)

Molecule IDs in data.reline (from build_data.py):
  1..10   choline
  11..20  chloride
  21..40  urea
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats


# Molecule-ID ranges written by scripts/build_data.py
CHOLINE_MOLS = list(range(1, 11))
CHLORIDE_MOLS = list(range(11, 21))
UREA_MOLS = list(range(21, 41))

SPECIES_MOLS = {
    "choline": CHOLINE_MOLS,
    "chloride": CHLORIDE_MOLS,
    "urea": UREA_MOLS,
}

# Pair definitions for mixture s2 / RDF plots (label, species_i, species_j, like_like)
COM_RDF_PAIRS = [
    ("nn", "choline", "choline", True),      # choline–choline (was N–N site proxy)
    ("clcl", "chloride", "chloride", True),
    ("uu", "urea", "urea", True),
    ("ncl", "choline", "chloride", False),
    ("nu", "choline", "urea", False),
    ("clu", "chloride", "urea", False),
]


def compute_molecular_coms(positions, masses, mol_ids):
    """
    Compute the center of mass of every molecule in one snapshot.

    Parameters
    ----------
    positions : (N, 3) array
        Atom coordinates (Å). Prefer *unwrapped* (xu,yu,zu) so a molecule that
        straddles a periodic boundary is not split across the box.
    masses : (N,) array
        Atomic masses (amu).
    mol_ids : (N,) array of int
        Molecule ID for each atom (column 2 in LAMMPS `Atoms # full`).

    Returns
    -------
    coms : dict[int, ndarray shape (3,)]
        Mapping molecule_id -> COM position (Å).
    """
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    mol_ids = np.asarray(mol_ids, dtype=int)
    if positions.shape[0] != masses.shape[0] or positions.shape[0] != mol_ids.shape[0]:
        raise ValueError("positions, masses, and mol_ids must have the same length")

    # Accumulators: sum(m*r) and sum(m) per molecule
    weighted = defaultdict(lambda: np.zeros(3, dtype=float))
    mass_sum = defaultdict(float)
    for r, m, mid in zip(positions, masses, mol_ids):
        weighted[mid] += m * r
        mass_sum[mid] += m

    coms = {}
    for mid, mr in weighted.items():
        total_m = mass_sum[mid]
        if total_m <= 0:
            raise ValueError(f"molecule {mid} has non-positive total mass")
        coms[mid] = mr / total_m
    return coms


def wrap_into_box(r, box_length):
    """Wrap a point into [0, L) for a cubic box from 0 to L."""
    L = float(box_length)
    return np.mod(r, L)


def minimum_image_distance(ri, rj, box_length):
    """Distance between two points with cubic periodic boundaries."""
    dr = np.asarray(rj, dtype=float) - np.asarray(ri, dtype=float)
    L = float(box_length)
    dr -= L * np.round(dr / L)
    return float(np.linalg.norm(dr))


def parse_lammps_dump(path: Path):
    """
    Parse a LAMMPS dump custom file with columns:
      id mol type mass xu yu zu

    Yields dicts:
      step, box_length (cubic), ids, mols, types, masses, coords (N,3) unwrapped
    """
    path = Path(path)
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline().split()[0])
            assert f.readline().startswith("ITEM: NUMBER OF ATOMS")
            n_atoms = int(f.readline().split()[0])
            box_header = f.readline()
            assert box_header.startswith("ITEM: BOX BOUNDS")
            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            zlo, zhi = map(float, f.readline().split()[:2])
            # Prefer x-edge as cubic length (pipeline builds cubes)
            box_length = xhi - xlo
            atoms_header = f.readline()
            assert atoms_header.startswith("ITEM: ATOMS")
            # Expected: id mol type mass xu yu zu
            cols = atoms_header.split()[2:]
            need = ["id", "mol", "type", "mass", "xu", "yu", "zu"]
            if cols[:7] != need:
                raise ValueError(
                    f"Unexpected dump columns {cols}; expected {need} ..."
                )
            ids = np.empty(n_atoms, dtype=int)
            mols = np.empty(n_atoms, dtype=int)
            types = np.empty(n_atoms, dtype=int)
            masses = np.empty(n_atoms, dtype=float)
            coords = np.empty((n_atoms, 3), dtype=float)
            for i in range(n_atoms):
                parts = f.readline().split()
                ids[i] = int(parts[0])
                mols[i] = int(parts[1])
                types[i] = int(parts[2])
                masses[i] = float(parts[3])
                coords[i, 0] = float(parts[4])
                coords[i, 1] = float(parts[5])
                coords[i, 2] = float(parts[6])
            yield {
                "step": step,
                "box_length": box_length,
                "ids": ids,
                "mols": mols,
                "types": types,
                "masses": masses,
                "coords": coords,
            }


def trajectory_molecular_coms(dump_path: Path):
    """
    For every dump frame, compute COM of every molecule.

    Returns
    -------
    times_fs : (F,)  timestep * dt with dt=1 fs in this project
    box_length : float
    com_traj : dict[mol_id] -> (F, 3) unwrapped COM trajectory
    """
    times = []
    com_traj = None
    box_length = None
    for frame in parse_lammps_dump(dump_path):
        times.append(float(frame["step"]))  # dt = 1 fs → time_fs = step
        box_length = frame["box_length"]
        coms = compute_molecular_coms(frame["coords"], frame["masses"], frame["mols"])
        if com_traj is None:
            com_traj = {mid: [] for mid in coms}
        for mid, r in coms.items():
            com_traj[mid].append(r)
    if not times:
        raise ValueError(f"no frames found in {dump_path}")
    for mid in com_traj:
        com_traj[mid] = np.asarray(com_traj[mid], dtype=float)
    return np.asarray(times, dtype=float), float(box_length), com_traj


def species_msd_from_coms(times_fs, com_traj, mol_ids):
    """
    Einstein MSD for one species: average |R(t)-R(0)|^2 over its molecules.
    Uses unwrapped COMs. Returns times, msd_values, D (Å²/fs), R2_late.
    """
    if not mol_ids:
        return times_fs, np.full_like(times_fs, np.nan), float("nan"), 0.0
    msd = np.zeros(len(times_fs), dtype=float)
    n = 0
    for mid in mol_ids:
        if mid not in com_traj:
            continue
        traj = com_traj[mid]
        dr = traj - traj[0]
        msd += np.sum(dr * dr, axis=1)
        n += 1
    if n == 0:
        return times_fs, np.full_like(times_fs, np.nan), float("nan"), 0.0
    msd /= n
    # Late-time Einstein fit
    start = int(len(times_fs) * 0.4)
    if len(times_fs) - start < 3:
        return times_fs, msd, float("nan"), 0.0
    slope, _intercept, r, _p, _se = stats.linregress(times_fs[start:], msd[start:])
    D = slope / 6.0
    return times_fs, msd, float(D), float(r**2)


def partial_rdf_coms(com_traj, mols_i, mols_j, box_length, n_bins=200, same_species=False):
    """
    Histogram g(r) between COMs of two species from the *last* frame
    (and optionally average over frames — here we average over all frames for
    better statistics since dump cadence is coarse).

    Returns r_centers (Å), g(r).
    """
    L = float(box_length)
    r_max = 0.5 * L
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins, dtype=float)
    n_frames = 0

    # Number of frames from any trajectory length
    sample_mid = next(iter(com_traj))
    n_total_frames = com_traj[sample_mid].shape[0]

    rho_j = len(mols_j) / (L**3)

    for f in range(n_total_frames):
        pos_i = [wrap_into_box(com_traj[m][f], L) for m in mols_i if m in com_traj]
        pos_j = [wrap_into_box(com_traj[m][f], L) for m in mols_j if m in com_traj]
        if not pos_i or not pos_j:
            continue
        pos_i = np.asarray(pos_i)
        pos_j = np.asarray(pos_j)
        for a, ri in enumerate(pos_i):
            for b, rj in enumerate(pos_j):
                if same_species and b <= a:
                    continue
                d = minimum_image_distance(ri, rj, L)
                if d < r_max:
                    bin_idx = int(d / r_max * n_bins)
                    if bin_idx >= n_bins:
                        bin_idx = n_bins - 1
                    hist[bin_idx] += 1.0
        n_frames += 1

    if n_frames == 0:
        r = 0.5 * (edges[:-1] + edges[1:])
        return r, np.ones_like(r)

    # Ideal-gas shell counts for normalization
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    n_i = len(mols_i)
    if same_species:
        # Each unique pair counted once; expected count per frame = N*(N-1)/2 * (shell/V) / (something)
        # Standard: g = hist / (n_frames * N_i * rho_j * shell_vol) but for same species
        # each pair counted once so factor is N_i * (N_i-1)/2 vs N_i * N_j with j=i.
        # Using: average number of j neighbors in shell around each i:
        # hist_total / (n_frames * N_i)  compared to rho * shell_vol,
        # but we skipped b<=a so we counted each pair once → multiply hist by 2 / N_i effectively
        # Equivalent normalization used widely:
        expected = n_frames * n_i * (n_i - 1) / 2.0 * (shell_vol / (L**3))
        # which is n_frames * [N choose 2] * (shell_vol / V)
        g = np.divide(hist, expected, out=np.zeros_like(hist), where=expected > 0)
    else:
        expected = n_frames * n_i * rho_j * shell_vol
        g = np.divide(hist, expected, out=np.zeros_like(hist), where=expected > 0)
    return r, g


def write_msd_dat(path: Path, times_fs, msd):
    """Write two-column MSD file compatible with existing plotters."""
    with open(path, "w") as f:
        f.write("# time_fs msd  (molecular COM)\n")
        for t, m in zip(times_fs, msd):
            f.write(f"{t:.6g} {m:.8e}\n")


def write_rdf_pair_dat(path: Path, r, g):
    """Write a simple r, g(r) table (last averaged COM RDF)."""
    with open(path, "w") as f:
        f.write("# r_Ang  g_com\n")
        for ri, gi in zip(r, g):
            f.write(f"{ri:.6f} {gi:.8e}\n")
