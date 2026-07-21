"""
Drive reline DES simulations: prepare run dir, run LAMMPS, parse MSD/RDF.

Rosenfeld-style analysis uses:
  - molecular-site diffusion (choline N, Cl, urea C) instead of all-atom MSD
  - multicomponent pair excess entropy from partial site-site RDFs
"""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "templates" / "in.des.template"
DATA_FILE = ROOT / "data" / "data.reline"
OUTPUT_DIR = ROOT / "output"
PACKMOL_TEMPLATE = ROOT / "packmol" / "pack_reline.inp.template"
PACKMOL_INP = ROOT / "packmol" / "pack_reline.inp"

# ---------------------------------------------------------------------------
# State-point sweep
# WHY wider T window: Rosenfeld needs s2 to change; 300-304 K was too narrow.
# WHY denser around 1.14-1.23: experimental anhydrous reline is ~1.16-1.20 g/cm3
# near room temperature; 1.30-1.35 was unrealistically compressed.
# WHY more steps (NOT a bigger timestep): physical time = steps × dt.
# We keep dt = 1 fs (safe for all-atom H vibrations). Making dt larger without
# bond constraints (SHAKE) risks unstable MD. Longer runs = more steps.
# Target: ~1 ns production so viscous Reline can leave the caging regime.
# ---------------------------------------------------------------------------
temp_start, temp_end, temp_step = 298.0, 348.0, 10.0   # 298,308,318,328,338 K
density_start, density_end, density_step = 1.14, 1.24, 0.03  # 1.14,1.17,1.20,1.23
eq_steps = 200000       # 200 ps equilibration  (dt=1 fs)
prod_steps = 1000000    # 1.0 ns production     (dt=1 fs)

# System counts (must match Packmol / build_data.py)
N_CHOLINE = 10
N_CHLORIDE = 10
N_UREA = 20
N_ATOMS = 380
N_MOLECULES = N_CHOLINE + N_CHLORIDE + N_UREA  # 40
# Mole fractions for molecular sites used in s2
X_CH = N_CHOLINE / N_MOLECULES   # 0.25
X_CL = N_CHLORIDE / N_MOLECULES  # 0.25
X_UR = N_UREA / N_MOLECULES      # 0.50

total_mass_amu = 2597.4  # 10 reline units (259.7 amu each)
reline_density = 1.2  # g/cm^3 reference
K_B = 1.380649e-23          # J/K
AMU_TO_KG = 1.660539e-27    # kg/amu
ANG2_FS_TO_M2_S = 1.0e-5    # 1 Å²/fs = 1e-5 m²/s

# Partial RDF files written by in.des.template (site-site)
# (filename, x_i, x_j, like_like) — cross terms counted once so factor 2 in s2
RDF_PARTIALS = [
    ("rdf_nn.out", X_CH, X_CH, True),
    ("rdf_clcl.out", X_CL, X_CL, True),
    ("rdf_uu.out", X_UR, X_UR, True),
    ("rdf_ncl.out", X_CH, X_CL, False),
    ("rdf_nu.out", X_CH, X_UR, False),
    ("rdf_clu.out", X_CL, X_UR, False),
]

MSD_FILES = [
    ("msd_choline.dat", "choline"),
    ("msd_chloride.dat", "chloride"),
    ("msd_urea.dat", "urea"),
]


def _find_exe(names):
    """Return first executable found on PATH from names."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def run_packmol(box_length: float):
    """Fill Packmol template with box length and run packmol."""
    packmol = _find_exe(["packmol", "packmol.x"])
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")

    text = PACKMOL_TEMPLATE.read_text()
    text = text.replace("BOX_PLACEHOLDER", f"{box_length:.6f}")
    PACKMOL_INP.write_text(text)

    with open(ROOT / "packmol" / "packmol.log", "w") as log:
        subprocess.run(
            [packmol],
            stdin=open(PACKMOL_INP, "r"),
            cwd=ROOT / "packmol",
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def run_build_data(box_length: float):
    """Rebuild data/data.reline for this box length."""
    # WHY sys.executable: bare "python" can be an old interpreter that cannot
    # parse type hints in build_data.py. Use the same interpreter as this pipeline.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_data.py"), str(box_length)],
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
    lmp = _find_exe(["lmp", "lmp_serial", "lammps", "lmp_mpi"])
    if lmp is None:
        raise RuntimeError("LAMMPS executable not found on PATH (tried lmp, lmp_serial, lammps, lmp_mpi)")

    mpirun = _find_exe(["mpirun", "mpiexec"])
    with open(run_dir / "experiment.log", "w") as log:
        if mpirun:
            cmd = [mpirun, "-np", "4", lmp, "-in", "in.des"]
        else:
            cmd = [lmp, "-in", "in.des"]
        subprocess.run(
            cmd,
            cwd=run_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def parse_msd(path: Path):
    """Read two-column MSD file: time_fs, msd."""
    times, msd = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            times.append(float(parts[0]))
            msd.append(float(parts[1]))
    return np.array(times), np.array(msd)


def compute_diffusion(times, msd, late_frac_start=0.4):
    """
    Einstein: MSD = 6 D t  →  D = slope/6.
    Uses the late-time window and returns (D, R2_late) so callers can judge quality.
    """
    if len(times) < 5:
        return float("nan"), 0.0
    start = int(len(times) * late_frac_start)
    t = times[start:]
    m = msd[start:]
    slope, _intercept, r, _p, _se = stats.linregress(t, m)
    return slope / 6.0, float(r**2)  # Å^2 / fs


def parse_rdf(path: Path):
    """Return last time-averaged RDF block: r (Å), g(r)."""
    r, g = [], []
    with open(path) as f:
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


def pair_integrand_integral(r, g):
    """∫ [g ln g - g + 1] r^2 dr  (g=0 bins contribute +1)."""
    integrand = np.ones_like(g)
    mask = g > 0
    integrand[mask] = g[mask] * np.log(g[mask]) - g[mask] + 1.0
    return float(np.trapezoid(integrand * r**2, r))


def truncate_rdf_to_half_box(r, g, box_length):
    """
    Keep only r < L/2.

    WHY: with periodic boundaries, distances beyond half the box are not unique
    (you are seeing the molecule's periodic image). Integrating g(r) past L/2
    adds noise, not real liquid structure. Increasing the pair cutoff cannot
    fix this — you need a larger box (more molecules).
    """
    rmax = 0.5 * box_length
    keep = r < rmax
    return r[keep], g[keep], rmax


def compute_s2_mixture(run_dir: Path, number_density_mol, box_length=None):
    """
    Multicomponent pair excess-entropy proxy (per molecule, /kB):

      s2/kB = -2 π ρ_mol Σ_i Σ_j x_i x_j ∫ [g_ij ln g_ij - g_ij + 1] r^2 dr

    Like-like pairs appear once (i=j). Cross pairs are stored once, so we
    multiply by 2 to account for both (i,j) and (j,i) in the double sum.

    WHY: this is the standard mixture extension of the atomic s2 formula and
    matches molecular mole fractions of our site RDFs.
    """
    s2 = 0.0
    diagnostics = {}
    for fname, xi, xj, like_like in RDF_PARTIALS:
        path = run_dir / fname
        r, g = parse_rdf(path)
        if box_length is not None and len(r):
            r, g, rmax = truncate_rdf_to_half_box(r, g, box_length)
        else:
            rmax = float(r[-1]) if len(r) else float("nan")
        I = pair_integrand_integral(r, g)
        weight = xi * xj if like_like else 2.0 * xi * xj
        s2 += -2.0 * np.pi * number_density_mol * weight * I
        g_max = float(g.max()) if len(g) else float("nan")
        g_tail = float(g[-10:].mean()) if len(g) >= 10 else float("nan")
        diagnostics[fname] = {
            "g_max": g_max,
            "g_tail": g_tail,
            "integral": I,
            "rmax_used": rmax,
        }
    return float(s2), diagnostics


def reduce_diffusion(D_ang2_fs, temperature_K, number_density_per_ang3, mass_per_particle_amu):
    """
    D* = D * rho_n^(1/3) / sqrt(kT/m)
    Returns dimensionless D*.
    """
    D_m2_s = D_ang2_fs * ANG2_FS_TO_M2_S
    rho_n_m3 = number_density_per_ang3 * 1.0e30   # 1/Å³ → 1/m³
    m_kg = mass_per_particle_amu * AMU_TO_KG
    thermal_speed = np.sqrt(K_B * temperature_K / m_kg)  # m/s
    return D_m2_s * (rho_n_m3 ** (1.0 / 3.0)) / thermal_speed


def analyze_state_point(run_dir: Path, temperature: float, density: float):
    """
    Parse species MSDs + partial RDFs for one finished state point.
    Returns dict with D (avg), D*, s2, and quality diagnostics.
    """
    L = box_length_angstrom(density, total_mass_amu)
    rho_mol = N_MOLECULES / (L**3)
    mass_per_mol = total_mass_amu / N_MOLECULES

    species_D = {}
    species_R2 = {}
    for fname, label in MSD_FILES:
        times, msd = parse_msd(run_dir / fname)
        D, r2 = compute_diffusion(times, msd)
        species_D[label] = D
        species_R2[label] = r2

    # Average only positive, finite species diffusivities
    good = [D for D in species_D.values() if np.isfinite(D) and D > 0]
    D_avg = float(np.mean(good)) if good else float("nan")
    D_star = (
        reduce_diffusion(D_avg, temperature, rho_mol, mass_per_mol)
        if np.isfinite(D_avg) and D_avg > 0
        else float("nan")
    )

    # Cap s2 integral at L/2 (periodic-boundary limit), not the force cutoff.
    s2, rdf_diag = compute_s2_mixture(run_dir, rho_mol, box_length=L)

    # Liquid-like RDF check: at least one partial g_max should be clearly > 1
    g_maxes = [d["g_max"] for d in rdf_diag.values()]
    max_gmax = max(g_maxes) if g_maxes else float("nan")

    return {
        "D": D_avg,
        "Dstar": D_star,
        "s2": s2,
        "species_D": species_D,
        "species_R2": species_R2,
        "rdf_diag": rdf_diag,
        "max_gmax": max_gmax,
        "rho_mol": rho_mol,
        "box_length": L,
    }


def save_state_msd_plot(run_dir: Path):
    """
    Write run_dir/msd_check.png for this one state point.

    WHY per-folder plots: easier to open T*_rho*/ and see whether THAT point
    reached a linear Einstein regime before trusting its D*.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharex=True)
    for col, (fname, label) in enumerate(MSD_FILES):
        ax = axes[col]
        path = run_dir / fname
        if not path.exists():
            ax.set_title(f"{label}: missing")
            continue
        t, m = parse_msd(path)
        ax.plot(t, m, color="0.3", lw=1.0)
        start = int(len(t) * 0.4)
        if len(t) > start + 2:
            slope, intercept, r, *_ = stats.linregress(t[start:], m[start:])
            t_fit = t[start:]
            ax.plot(
                t_fit,
                slope * t_fit + intercept,
                color="#FF9999",
                lw=1.5,
                label=f"late fit R²={r**2:.3f}\nD={slope/6:.2e} Å²/fs",
            )
            ax.legend(fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("time (fs)")
        if col == 0:
            ax.set_ylabel("MSD (Å²)")
    fig.suptitle(f"MSD check — {run_dir.name}", fontsize=11)
    fig.tight_layout()
    out = run_dir / "msd_check.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def save_state_rdf_plot(run_dir: Path, box_length: float):
    """
    Write run_dir/rdf_check.png for this one state point.

    Marks L/2 so you can see where periodic boundaries make g(r) unreliable.
    """
    pair_files = [p[0] for p in RDF_PARTIALS]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0), sharex=True)
    axes = axes.ravel()
    half = 0.5 * box_length
    for col, fname in enumerate(pair_files):
        ax = axes[col]
        path = run_dir / fname
        if not path.exists():
            ax.set_title(f"{fname}: missing")
            continue
        r, g = parse_rdf(path)
        ax.plot(r, g, lw=1.0, color="0.25")
        ax.axhline(1.0, color="0.6", ls="--", lw=0.7)
        ax.axvline(half, color="#c44e52", ls=":", lw=1.2, label=f"L/2={half:.2f} Å")
        # shade unreliable region beyond half-box
        if len(r):
            ax.axvspan(half, float(r[-1]), color="#c44e52", alpha=0.08)
        g_in = g[r < half] if len(r) else g
        gmax = float(g_in.max()) if len(g_in) else float("nan")
        ax.set_title(f"{fname.replace('.out', '')}\nmax(<L/2)={gmax:.2f}", fontsize=8)
        ax.set_xlabel("r (Å)")
        if col % 3 == 0:
            ax.set_ylabel("g(r)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Partial RDFs — {run_dir.name} (shaded = beyond L/2, do not trust)",
        fontsize=11,
    )
    fig.tight_layout()
    out = run_dir / "rdf_check.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def save_msd_diagnostic(path_run: Path, sample_dirs):
    """
    Overview MSD figure for a few state points (also written under path_run/).

    WHY: a Rosenfeld fit is meaningless if D itself comes from a noisy non-linear MSD.
    """
    n = len(sample_dirs)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])
    for row, run_dir in enumerate(sample_dirs):
        for col, (fname, label) in enumerate(MSD_FILES):
            ax = axes[row, col]
            path = run_dir / fname
            if not path.exists():
                ax.set_title(f"{run_dir.name}\n{label}: missing")
                continue
            t, m = parse_msd(path)
            ax.plot(t, m, color="0.3", lw=1.0)
            start = int(len(t) * 0.4)
            if len(t) > start + 2:
                slope, intercept, r, *_ = stats.linregress(t[start:], m[start:])
                t_fit = t[start:]
                ax.plot(t_fit, slope * t_fit + intercept, color="#FF9999", lw=1.5,
                        label=f"late fit R²={r**2:.3f}")
                ax.legend(fontsize=7)
            ax.set_title(f"{run_dir.name}\n{label}", fontsize=8)
            if row == n - 1:
                ax.set_xlabel("time (fs)")
            if col == 0:
                ax.set_ylabel("MSD (Å²)")
    fig.suptitle("MSD linearity check (late-time Einstein window)", fontsize=11)
    fig.tight_layout()
    out = path_run / "msd_check.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Wrote", out)


def save_rdf_diagnostic(path_run: Path, sample_dirs, box_lengths=None):
    """Overview partial g(r) figure for a few state points under path_run/."""
    n = len(sample_dirs)
    if n == 0:
        return
    pair_files = [p[0] for p in RDF_PARTIALS]
    fig, axes = plt.subplots(n, len(pair_files), figsize=(14, 2.4 * n), sharex=True)
    if n == 1:
        axes = np.array([axes])
    for row, run_dir in enumerate(sample_dirs):
        half = None
        if box_lengths is not None and row < len(box_lengths):
            half = 0.5 * box_lengths[row]
        for col, fname in enumerate(pair_files):
            ax = axes[row, col]
            path = run_dir / fname
            if not path.exists():
                continue
            r, g = parse_rdf(path)
            ax.plot(r, g, lw=1.0)
            ax.axhline(1.0, color="0.6", ls="--", lw=0.7)
            if half is not None:
                ax.axvline(half, color="#c44e52", ls=":", lw=1.0)
                if len(r):
                    ax.axvspan(half, float(r[-1]), color="#c44e52", alpha=0.08)
                g_in = g[r < half]
                gmax = float(g_in.max()) if len(g_in) else float("nan")
            else:
                gmax = float(g.max()) if len(g) else float("nan")
            ax.set_title(f"{run_dir.name}\n{fname.replace('.out','')}\nmax={gmax:.2f}", fontsize=7)
            if row == n - 1:
                ax.set_xlabel("r (Å)")
            if col == 0:
                ax.set_ylabel("g(r)")
    fig.suptitle(
        "Partial site-site RDFs (liquid-like peaks >> 1; red = beyond L/2)",
        fontsize=11,
    )
    fig.tight_layout()
    out = path_run / "rdf_check.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Wrote", out)


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
    print(f"Temps (K): {temps}")
    print(f"Densities (g/cm3): {dens}")
    print(f"eq_steps={eq_steps}, prod_steps={prod_steps}")
    print(f"State points: {len(temps) * len(dens)}")

    results_T, results_rho, results_D, results_Dstar, results_s2 = [], [], [], [], []
    results_meta = []
    summary_path = path_run / "summary.dat"
    finished_dirs = []

    for T in temps:
        for rho in dens:
            name = f"T{T:.1f}_rho{rho:.3f}"
            run_dir = prepare_run_dir(path_run, name, T, rho, eq_steps, prod_steps)
            run_lammps(run_dir)
            finished_dirs.append(run_dir)

            ana = analyze_state_point(run_dir, T, rho)
            # Per-state diagnostic PNGs inside T*_rho*/ (in addition to run-level overview)
            msd_png = save_state_msd_plot(run_dir)
            rdf_png = save_state_rdf_plot(run_dir, ana["box_length"])
            print(f"  wrote {msd_png.name}, {rdf_png.name}")

            results_T.append(T)
            results_rho.append(rho)
            results_D.append(ana["D"])
            results_Dstar.append(ana["Dstar"])
            results_s2.append(ana["s2"])
            results_meta.append(ana)
            print(
                f"{name}: D*={ana['Dstar']:.6e}, s2/kB={ana['s2']:.4f}, "
                f"MSD R2={{{', '.join(f'{k}:{v:.2f}' for k,v in ana['species_R2'].items())}}}, "
                f"max g(r)={ana['max_gmax']:.2f}"
            )

    with open(summary_path, "w") as f:
        f.write(
            "T_K    rho_g_cm3    D_Ang2_fs    Dstar    s2_per_kB    "
            "D_chol    D_cl    D_urea    R2_chol    R2_cl    R2_urea    max_gmax\n"
        )
        for T, rho, D, Dstar, s2, meta in zip(
            results_T, results_rho, results_D, results_Dstar, results_s2, results_meta
        ):
            sd = meta["species_D"]
            sr = meta["species_R2"]
            f.write(
                f"{T:.3f}  {rho:.5f}  {D:.8e}  {Dstar:.8e}  {s2:.6f}  "
                f"{sd['choline']:.8e}  {sd['chloride']:.8e}  {sd['urea']:.8e}  "
                f"{sr['choline']:.4f}  {sr['chloride']:.4f}  {sr['urea']:.4f}  "
                f"{meta['max_gmax']:.3f}\n"
            )
    print("Wrote", summary_path)

    summary_csv = path_run / "summary.csv"
    with open(summary_csv, "w") as f:
        f.write(
            "T_K,rho_g_cm3,D_Ang2_fs,Dstar,s2_per_kB,"
            "D_chol,D_cl,D_urea,R2_chol,R2_cl,R2_urea,max_gmax\n"
        )
        for T, rho, D, Dstar, s2, meta in zip(
            results_T, results_rho, results_D, results_Dstar, results_s2, results_meta
        ):
            sd = meta["species_D"]
            sr = meta["species_R2"]
            f.write(
                f"{T:.3f},{rho:.5f},{D:.8e},{Dstar:.8e},{s2:.6f},"
                f"{sd['choline']:.8e},{sd['chloride']:.8e},{sd['urea']:.8e},"
                f"{sr['choline']:.4f},{sr['chloride']:.4f},{sr['urea']:.4f},"
                f"{meta['max_gmax']:.3f}\n"
            )
    print("Wrote", summary_csv)

    # Run-level overview diagnostics on a few representative finished points
    sample = []
    sample_L = []
    if finished_dirs:
        idxs = [0, len(finished_dirs) // 2, len(finished_dirs) - 1]
        seen = set()
        for i in idxs:
            if i in seen:
                continue
            seen.add(i)
            sample.append(finished_dirs[i])
            sample_L.append(results_meta[i]["box_length"])
    save_msd_diagnostic(path_run, sample)
    save_rdf_diagnostic(path_run, sample, box_lengths=sample_L)

    plot_rosenfeld(path_run, results_s2, results_Dstar, results_T, results_rho)


def _legend_loc_least_overlap(ax, x, y):
    """
    Pick a legend corner that covers the fewest data points.

    Points are mapped into axes fraction coordinates [0,1]x[0,1]. Each corner
    owns a rectangle (about 40% width x 45% height); we choose the corner whose
    rectangle contains the fewest points. Ties prefer lower right, then lower
    left (usually emptier on Rosenfeld plots).
    """
    if len(x) == 0:
        return "lower right"

    ax.figure.canvas.draw()
    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    disp = ax.transData.transform(pts)
    axes_pts = ax.transAxes.inverted().transform(disp)

    corners = [
        ("lower right", 0.55, 1.0, 0.0, 0.48),
        ("lower left", 0.0, 0.45, 0.0, 0.48),
        ("upper right", 0.55, 1.0, 0.52, 1.0),
        ("upper left", 0.0, 0.45, 0.52, 1.0),
    ]
    best_loc, best_count = "lower right", np.inf
    for loc, xmin, xmax, ymin, ymax in corners:
        inside = (
            (axes_pts[:, 0] >= xmin)
            & (axes_pts[:, 0] <= xmax)
            & (axes_pts[:, 1] >= ymin)
            & (axes_pts[:, 1] <= ymax)
        )
        count = int(np.count_nonzero(inside))
        if count < best_count:
            best_count = count
            best_loc = loc
    return best_loc


def plot_rosenfeld(path_run: Path, s2s, Dstars, Ts, rhos):
    """Rosenfeld plot: color = T, marker shape = density; combined legend."""
    s2s = np.asarray(s2s, dtype=float)
    Dstars = np.asarray(Dstars, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    rhos = np.asarray(rhos, dtype=float)

    positive = np.isfinite(Dstars) & (Dstars > 0) & np.isfinite(s2s)
    n_skipped = int(np.sum(~positive))
    if n_skipped:
        print(f"Warning: skipping {n_skipped} non-positive/non-finite D* point(s) in Rosenfeld plot/fit")

    s2_plot = s2s[positive]
    D_plot = Dstars[positive]
    T_plot = Ts[positive]
    rho_plot = rhos[positive]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.minorticks_on()
    ax.set_yscale("log")

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    unique_rhos = np.unique(rho_plot)
    cmap = plt.cm.viridis
    if len(T_plot) == 0:
        t_min, t_max = 0.0, 1.0
    elif np.isclose(T_plot.min(), T_plot.max()):
        t_min, t_max = float(T_plot.min()) - 1.0, float(T_plot.max()) + 1.0
    else:
        t_min, t_max = float(T_plot.min()), float(T_plot.max())
    norm = plt.Normalize(vmin=t_min, vmax=t_max)

    for i, rho in enumerate(unique_rhos):
        mask = np.isclose(rho_plot, rho)
        ax.scatter(
            s2_plot[mask],
            D_plot[mask],
            c=T_plot[mask],
            cmap=cmap,
            norm=norm,
            marker=markers[i % len(markers)],
            s=55,
            alpha=0.85,
            edgecolors="0.2",
            linewidths=0.4,
            label=rf"$\rho$ = {rho:.2f} g/cm$^3$",
            zorder=3,
        )

    if len(s2_plot) > 1:
        slope, intercept, r, _p, _se = stats.linregress(s2_plot, np.log(D_plot))
        r_squared = r**2
        order = np.argsort(s2_plot)
        x_line = s2_plot[order]
        fit_label = (
            f"$D^* = \\exp({slope:.3f}\\, s_2/k_B + {intercept:.3f})$\n"
            f"$R^2 = {r_squared:.4f}$"
        )
        ax.plot(
            x_line,
            np.exp(slope * x_line + intercept),
            color="#FF9999",
            linewidth=2,
            label=fit_label,
            zorder=2,
        )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    cbar.set_label("T (K)")

    legend_loc = _legend_loc_least_overlap(ax, s2_plot, D_plot)
    ax.legend(loc=legend_loc, fontsize=8, framealpha=0.9)
    ax.set_xlabel(r"$s_2/k_B$ (molecular-site mixture)")
    ax.set_ylabel(r"$D^*$ (avg molecular-site)")
    ax.set_title("Reduced Diffusion vs. Pairwise Excess Entropy")
    fig.tight_layout()
    out = path_run / "plot.png"
    fig.savefig(out, dpi=150)
    print("Wrote", out, f"(legend at {legend_loc})")
    plt.close(fig)


if __name__ == "__main__":
    main()
