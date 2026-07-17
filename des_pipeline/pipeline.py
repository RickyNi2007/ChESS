"""
Drive reline DES simulations: prepare run dir, run LAMMPS, parse MSD/RDF.
"""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "in.des.template"
DATA_FILE = ROOT / "data" / "data.reline"
OUTPUT_DIR = ROOT / "output"
PACKMOL_TEMPLATE = ROOT / "packmol" / "pack_reline.inp.template"
PACKMOL_INP = ROOT / "packmol" / "pack_reline.inp"

#constants
temp_start, temp_end, temp_step = 298.0, 348.0, 10.0
density_start, density_end, density_step = 1.15, 1.35, 0.05
eq_steps = 5000
prod_steps = 10000
total_mass_amu = 2597.4 #10 relines (259.7 amu)
reline_density = 1.2 #g/cm^3
K_B = 1.380649e-23          # J/K
AMU_TO_KG = 1.660539e-27    # kg/amu
ANG2_FS_TO_M2_S = 1.0e-5    # 1 Å²/fs = 1e-5 m²/s


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


def prepare_run_dir(path_run: Path, run_name: str, temperature: float, density: float,
                    eq_steps: int = 5000, prod_steps: int = 10000) -> Path:
    """
    For one (T, density): compute L, pack, build data, create point folder under path_run.
    """
    L = box_length_angstrom(density, total_mass_amu)
    print(f"T={temperature}, rho={density} -> L={L:.4f} Angstrom")

    run_packmol(L)
    run_build_data(L)

    run_dir = path_run / run_name
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

def reduce_diffusion(D_ang2_fs, temperature_K, number_density_per_ang3, mass_per_atom_amu):
    """
    D* = D * rho_n^(1/3) / sqrt(kT/m)
    Returns dimensionless D*.
    """
    D_m2_s = D_ang2_fs * ANG2_FS_TO_M2_S
    rho_n_m3 = number_density_per_ang3 * 1.0e30   # 1/Å³ → 1/m³
    m_kg = mass_per_atom_amu * AMU_TO_KG
    thermal_speed = np.sqrt(K_B * temperature_K / m_kg)  # m/s
    return D_m2_s * (rho_n_m3 ** (1.0 / 3.0)) / thermal_speed

def main():
    if temp_start == temp_end:
        temps = np.array([temp_end])
    else:
        temps = np.arange(temp_start, temp_end, temp_step)
    if density_start == density_end:
        dens = np.array([density_end])
    else:
        dens = np.arange(density_start, density_end, density_step)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path_run = OUTPUT_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    path_run.mkdir(parents=True, exist_ok=True)
    print("Run folder:", path_run)

    results_T, results_rho, results_D, results_Dstar, results_s2 = [], [], [], [], []
    summary_path = path_run / "summary.dat"
    for T in temps:
        for rho in dens:
            name = f"T{T:.1f}_rho{rho:.3f}"
            run_dir = prepare_run_dir(path_run, name, T, rho, eq_steps, prod_steps)
            run_lammps(run_dir)
            times, msd = parse_msd(run_dir)
            D = compute_diffusion(times, msd)
            r, g = parse_rdf(run_dir)
            # N_atoms / L^3
            L = box_length_angstrom(rho, total_mass_amu)
            rho_n = 380.0 / (L**3)
            s2 = compute_s2(r, g, rho_n)
            D_star = reduce_diffusion(D, T, rho_n, total_mass_amu / 380.0)
            results_T.append(T)
            results_rho.append(rho)
            results_D.append(D)
            results_Dstar.append(D_star)
            results_s2.append(s2)
            print(f"D*={D_star:.6e}, s2/kB={s2:.4f}")
    with open(summary_path, "w") as f:
        f.write("T_K    rho_g_cm3    D_Ang2_fs    Dstar    s2_per_kB\n")
        for T, rho, D, Dstar, s2 in zip(
            results_T, results_rho, results_D, results_Dstar, results_s2
        ):
            f.write(f"{T:.3f}  {rho:.5f}  {D:.8e}  {Dstar:.8e}  {s2:.6f}\n")
    print("Wrote", summary_path)
    plot_rosenfeld(path_run, results_s2, results_Dstar, temps, dens)


def plot_rosenfeld(path_run: Path, s2s, Dstars, temps, dens):
    s2s = np.asarray(s2s, dtype=float)
    Dstars = np.asarray(Dstars, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.minorticks_on()
    ax.set_yscale("log")
    ax.scatter(s2s, Dstars, alpha=0.7, s=50, color="black")

    if len(s2s) > 1 and np.all(Dstars > 0):
        # linregress on log(D*) so the fit is linear in Rosenfeld coordinates
        slope, intercept, r, _p, _se = stats.linregress(s2s, np.log(Dstars))
        r_squared = r**2
        order = np.argsort(s2s)
        x_line = s2s[order]
        ax.plot(x_line, np.exp(slope * x_line + intercept), color="#FF9999", linewidth=2)
        # Solved for reduced diffusion: D* = exp(a * s2/kB + b)
        info = (
            f"D* = exp({slope:.3f} · s₂/k_B + {intercept:.3f})\n"
            f"R² = {r_squared:.4f}\n"
            f"T={temps[0]:.0f}-{temps[-1]:.0f}     ΔT={temp_step:g}\n"
            f"ρ={dens[0]:.2f}-{dens[-1]:.2f}     Δρ={density_step:g}"
        )
        ax.text(
            0.58, 0.95, info,
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(r"$s_2/k_B$")
    ax.set_ylabel(r"$D^*$")
    ax.set_title("Reduced Diffusion vs. Pairwise Excess Entropy")
    fig.tight_layout()
    out = path_run / "plot.png"
    fig.savefig(out, dpi=150)
    print("Wrote", out)
    plt.close(fig)

if __name__ == "__main__":
    main()