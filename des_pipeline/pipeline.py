"""
Drive reline DES simulations: prepare run dir, run LAMMPS, parse MSD/RDF.
"""

from pathlib import Path
import shutil
import subprocess
import numpy as np


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "in.des.template"
DATA_FILE = ROOT / "data" / "data.reline"
OUTPUT_DIR = ROOT / "output"
PACKMOL_TEMPLATE = ROOT / "packmol" / "pack_reline.inp.template"
PACKMOL_INP = ROOT / "packmol" / "pack_reline.inp"

#constants
temp_start, temp_end, temp_step = 298.0, 299.0, 1
density_start, density_end, density_step = 1.2, 1.3, .1
eq_steps = 5000
prod_steps = 10000
total_mass_amu = 2597.4 #10 relines (259.7 amu)
reline_density = 1.2 #g/cm^3

def run_packmol(box_length: float):
    """Fill Packmol template with box length and run packmol."""
    text = PACKMOL_TEMPLATE.read_text()
    text = text.replace("BOX_PLACEHOLDER", f"{box_length:.6f}")
    PACKMOL_INP.write_text(text)

    # Run from packmol/ so relative paths in the inp work
    with open(ROOT / "packmol" / "packmol.log", "w") as log:
        subprocess.run(
            ["packmol"],
            stdin=open(PACKMOL_INP, "r"),
            cwd=ROOT / "packmol",
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def run_build_data(box_length: float):
    """Rebuild data/data.reline for this box length."""
    subprocess.run(
        ["python", str(ROOT / "scripts" / "build_data.py"), str(box_length)],
        cwd=ROOT,
        check=True,
    )


def write_lammps_input(run_dir: Path, temperature: float, eq_steps: int, prod_steps: int):
    """Fill in.des.template placeholders and write into run_dir."""
    text = TEMPLATE.read_text()
    text = text.replace("TEMP_PLACEHOLDER", str(temperature))
    text = text.replace("EQ_STEPS_PLACEHOLDER", str(eq_steps))
    text = text.replace("PROD_STEPS_PLACEHOLDER", str(prod_steps))
    (run_dir / "in.des").write_text(text)


def prepare_run_dir(run_name: str, temperature: float, density: float,
                    eq_steps: int = 5000, prod_steps: int = 10000) -> Path:
    """
    For one (T, density): compute L, pack, build data, create output run folder.
    """
    L = box_length_angstrom(density, total_mass_amu)
    print(f"T={temperature}, rho={density} -> L={L:.4f} Angstrom")

    run_packmol(L)
    run_build_data(L)

    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_FILE, run_dir / "data.reline")
    write_lammps_input(run_dir, temperature, eq_steps, prod_steps)
    return run_dir

def box_length_angstrom(mass_density_g_per_cm3, total_mass_amu):
    """
    total_mass_amu = sum of all atomic masses in the box (g/mol numerically = amu/atom summed).
    Returns cube side length in Angstroms.
    """
    AMU_TO_G = 1.66053906660e-24  # grams per amu
    m_grams = total_mass_amu * AMU_TO_G
    V_cm3 = m_grams / mass_density_g_per_cm3
    L_cm = V_cm3 ** (1.0 / 3.0)
    L_angstrom = L_cm * 1.0e8
    return L_angstrom

def run_lammps(run_dir: Path):
    """Run LAMMPS in run_dir; write experiment.log."""
    with open(run_dir / "experiment.log", "w") as log:
        subprocess.run(
            ["mpirun", "-np", "4", "lmp", "-in", "in.des"],
            cwd=run_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )  # ensure your computer has 4 processors

def parse_msd(run_dir: Path):
    times, msd = [], []
    with open(run_dir / "msd.dat") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            times.append(float(parts[0]))
            msd.append(float(parts[1]))
    return np.array(times), np.array(msd)


def compute_diffusion(times, msd):
    """Einstein: MSD = 6 D t  →  D = slope/6. Uses late-time window."""
    start = int(len(times) * 0.4)
    slope = np.polyfit(times[start:], msd[start:], 1)[0]
    return slope / 6.0  # Å^2 / fs  (for now; reduce later)


def parse_rdf(run_dir: Path):
    """Return last time-averaged RDF block: r (Å), g(r)."""
    r, g = [], []
    with open(run_dir / "rdf.out") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:  # "timestep nrows" header → new block
                r, g = [], []
                continue
            if len(parts) >= 3:
                r.append(float(parts[1]))
                g.append(float(parts[2]))
    return np.array(r), np.array(g)


def compute_s2(r, g, number_density):
    """
    Pair excess entropy proxy (per particle, /kB):
    s2/kB = -2 π ρ ∫ [g ln g - g + 1] r^2 dr
    number_density in atoms/Å^3
    """
    integrand = np.ones_like(g)
    mask = g > 0
    integrand[mask] = g[mask] * np.log(g[mask]) - g[mask] + 1.0
    return float(-2.0 * np.pi * number_density * np.trapezoid(integrand * r**2, r))

def main():
    if temp_start == temp_end:
        temps = [temp_end]
    else:
        temps = np.arange(temp_start, temp_end, temp_step)
    if density_start == density_end:
        dens = [density_end]
    else:
        dens = np.arange(density_start, density_end, density_step)
    for T in temps:
        for rho in dens:
            name = f"T{T:.1f}_rho{rho:.3f}"
            run_dir = prepare_run_dir(name, T, rho, eq_steps, prod_steps)
            run_lammps(run_dir)
            times, msd = parse_msd(run_dir)
            D = compute_diffusion(times, msd)
            r, g = parse_rdf(run_dir)
            # N_atoms / L^3
            L = box_length_angstrom(rho, total_mass_amu)
            rho_n = 380.0 / (L**3)
            s2 = compute_s2(r, g, rho_n)
            print(f"  D = {D:.6e} Ang^2/fs,  s2/kB = {s2:.4f}")

if __name__ == "__main__":
    main()