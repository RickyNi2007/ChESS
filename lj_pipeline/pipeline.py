import os 
import subprocess
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from scipy import stats
from datetime import datetime

#Density or Temperature Config
temp_start = .8
temp_end = 1.2 #doesn't include temp_end as a data point
temp_step_size = .1
density_start = .7
density_end = .85
density_step_size = .01

if  temp_start == temp_end and density_start == density_end: #constant temp & dens
    temp_change, density_change = False, False
    temperature = temp_start
    density = density_start
elif temp_start != temp_end and density_start != density_end: #changing temp & dens
    temp_change, density_change = True, True
    dens = np.arange(density_start, density_end, density_step_size)
    temps = np.arange(temp_start, temp_end, temp_step_size)
elif temp_start == temp_end :                    #constant temp, changing dens
    temp_change, density_change = False, True
    dens = np.arange(density_start, density_end, density_step_size)
    temperature = temp_start
else:                                           #constant dens, constant temp
    temp_change, density_change = True, False
    temps = np.arange(temp_start, temp_end, temp_step_size)
    density = density_start


#Run Config
run_length = 20000
num_bins = 200
N_every = 100
N_repeat = 200
N_freq = 20000
template_path = "templates/in.lj.template"
output_dir = "output"
lattice_reps = 8

#Computation Config
rosenfeld_paper = True
verlet_paper = False


def generate_lammps_input(T, rho, path_run):
    with open(template_path, "r") as file:
        file_string = file.read()
        file_string = file_string.replace("TEMP_PLACEHOLDER", str(T))
        file_string = file_string.replace("N_BINS_PLACEHOLDER", str(num_bins))
        file_string = file_string.replace("NEVERY_PLACEHOLDER", str(N_every))
        file_string = file_string.replace("NREPEAT_PLACEHOLDER", str(N_repeat))
        file_string = file_string.replace("NFREQ_PLACEHOLDER", str(N_freq))
        file_string = file_string.replace("RUN_LENGTH_PLACEHOLDER", str(run_length))
        file_string = file_string.replace("DENSITY_PLACEHOLDER", str(rho))
        file_string = file_string.replace("LATTICE_REPS_PLACEHOLDER", str(lattice_reps))
    if verlet_paper:
        path_to_output = os.path.join(path_run, f'T{T:.3f}_rho{rho:.3f}_VP')
    elif rosenfeld_paper: 
        path_to_output = os.path.join(path_run, f'T{T:.3f}_rho{rho:.3f}_RP')
    os.makedirs(path_to_output, exist_ok=True)
    with open(os.path.join(path_to_output, "in.lj"), "w") as file_out:
        file_out.write(file_string)
    return path_to_output

def run_simulations(sim_dir):
    with open(os.path.join(sim_dir, "experiment.log"), "w") as log_file:
        subprocess.run(['mpirun', '-np', '4', 'lmp', '-in', 'in.lj'], cwd=sim_dir, check=True, stdout=log_file)

def parse_msd(sim_dir):
    times = []
    msd = []
    with open(os.path.join(sim_dir, "msd.dat"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) != 2: continue
            try: 
                times.append(float(parts[0]))
                msd.append(float(parts[1]))
            except ValueError: continue
    times = np.array(times)
    msd = np.array(msd)
    return times, msd

def compute_diffusion(times, msd, temp, rho): 
    #splice off earlier parts of data 
    start_index = int(len(times)*0.4)
    t_fit = times[start_index:] 
    msd_fit = msd[start_index:] 
    coeffs = np.polyfit(t_fit, msd_fit, 1)
    diff_const = coeffs[0]
    if verlet_paper:
        return (diff_const/6)/(48**(1/2))
    if rosenfeld_paper: 
        return (diff_const/6)*rho**(1/3)*temp**(-1/2)
    return diff_const/6

def parse_rdf (sim_dir):
    r = []
    g = []
    with open(os.path.join(sim_dir, "rdf.out"), "r") as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) == 2: 
                r = []
                g = []
                continue
            r.append(float(parts[1]))
            g.append(float(parts[2]))
    return np.array(r), np.array(g)

def compute_s2(r,g, rho):
    integrand = np.ones_like(g)
    mask = g > 0
    integrand[mask] = (g[mask] * np.log(g[mask]) - g[mask] + 1) 
    integrand *= r**2
    integral = np.trapezoid(integrand, r)
    return -2*np.pi*rho*integral 

def parse_actual_temperature(sim_dir): #mean of NVE temps to output averaged temps
    temps = []
    with open(os.path.join(sim_dir, "thermo.dat"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) == 2:
                temps.append(float(parts[1]))
    return np.mean(temps) 

def parse_pressure(sim_dir): #parse and return pressure of system
    pressures = []
    with open(os.path.join(sim_dir, "pressure.dat"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 1:
                pressures.append(float(parts[1]))
    return np.mean(pressures)  

def parse_epair(sim_dir):
    vals = []
    with open(os.path.join(sim_dir, "pe.dat"), "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 1:
                vals.append(float(parts[1]))
    return np.mean(vals)/((lattice_reps**3)*4)

def run_pipeline():
    os.makedirs(output_dir, exist_ok=True)
    path_run = os.path.join(output_dir, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(path_run, exist_ok=True)
    actual_temps = []
    with open(os.path.join(path_run, "diff.dat"), 'w') as diff_dat:
        diff_dat.write("T        D            ρ\n")
        with open(os.path.join(path_run, "s2.dat"), 'w') as s2_dat:
            s2_dat.write("T        S2            ρ\n")
            #ADD IF TEMP CHANGE OR IF DENSITY CHANGE THEN MAKE CHANGES ACCORDINGLY
            if temp_change and density_change: 
                for t in temps: 
                    for rho in dens: 
                        path = generate_lammps_input(t, rho, path_run)
                        run_simulations(path)
                        time, msd = parse_msd(path)
                        D = compute_diffusion(time, msd, t, rho)
                        diff_dat.write(f"{t:.3f}    {D:.6f}     {rho:.3f}\n")
                        radius, g = parse_rdf(path)
                        s2 = compute_s2(radius, g, rho)
                        s2_dat.write(f"{t:.3f}    {s2:.6f}     {rho:.3f}\n")
                        t_actual = parse_actual_temperature(path)
                        actual_temps.append(t_actual)
                        pressure = parse_pressure(path)
                        epair = parse_epair(path)
                        print(f"T={t_actual:.3f}, rho={rho:.3f}, P={pressure:.6f}, D={D:.6f}, S2={s2:.6f}, E_pair={epair:.6f}")
            elif temp_change:
                for t in temps:
                    path = generate_lammps_input(t, density, path_run)
                    run_simulations(path)
                    time, msd = parse_msd(path)
                    D = compute_diffusion(time, msd, t, density)
                    diff_dat.write(f"{t:.3f}    {D:.6f}     {density:.3f}\n")
                    radius, g = parse_rdf(path)
                    s2 = compute_s2(radius, g, density)
                    s2_dat.write(f"{t:.3f}    {s2:.6f}     {density:.3f}\n")
                    t_actual = parse_actual_temperature(path)
                    actual_temps.append(t_actual)
                    pressure = parse_pressure(path)
                    epair = parse_epair(path)
                    print(f"T={t_actual:.3f}, rho={density:.3f}, P={pressure:.6f}, D={D:.6f}, S2={s2:.6f}, E_pair={epair:.6f}")
            elif density_change: 
                for rho in dens:
                    path = generate_lammps_input(temperature, rho, path_run)
                    run_simulations(path)
                    time, msd = parse_msd(path)
                    D = compute_diffusion(time, msd, temperature, rho)
                    diff_dat.write(f"{temperature:.3f}    {D:.6f}    {rho:.3f}\n")
                    radius, g = parse_rdf(path)
                    s2 = compute_s2(radius, g, rho)
                    s2_dat.write(f"{temperature:.3f}    {s2:.6f}    {rho:.3f}\n")
                    t_actual = parse_actual_temperature(path)
                    actual_temps.append(t_actual)
                    pressure = parse_pressure(path)
                    epair = parse_epair(path)
                    print(f"T={t_actual:.3f}, rho={rho:.3f}, P={pressure:.6f}, D={D:.6f}, S2={s2:.6f}, E_pair={epair:.6f}")
            else:
                path = generate_lammps_input(temperature, density, path_run)
                run_simulations(path)
                time, msd = parse_msd(path)
                D = compute_diffusion(time, msd, temperature, density)
                diff_dat.write(f"{temperature:.3f}    {D:.6f}    {density:.3f}\n")
                radius, g = parse_rdf(path)
                s2 = compute_s2(radius, g, density)
                s2_dat.write(f"{temperature:.3f}    {s2:.6f}    {density:.3f}\n")
                t_actual = parse_actual_temperature(path)
                actual_temps.append(t_actual)
                pressure = parse_pressure(path)
                epair = parse_epair(path)
                print(f"T={t_actual:.3f}, rho={density:.3f}, P={pressure:.6f}, D={D:.6f}, S2={s2:.6f}, E_pair={epair:.6f}")
    plot_graph(path_run, actual_temps)

def plot_graph(path, actual_temps):
    diff_coeffs = []
    S2 = []
    #compiling data
    with open(os.path.join(path, "diff.dat"), 'r') as diff_data:
        for line in diff_data:
            if line.startswith("T"): continue
            parts = line.split()
            diff_coeffs.append(float(parts[1]))
    with open(os.path.join(path, "s2.dat"), 'r') as s2_data:
        for line in s2_data:
            if line.startswith("T"): continue
            parts = line.split()
            S2.append(float(parts[1]))
    S2 = np.array(S2)
    S2 = -S2 #only for negation purposes to match graph
    diff_coeffs = np.array(diff_coeffs)

    #graphing
    fig, ax = plt.subplots()
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.minorticks_on()
    ax.set_yscale('log')
    ax.set_yticks(ax.get_yticks())
    ax.scatter(S2, diff_coeffs, alpha=0.7, s=50)
    #for i, T in enumerate(actual_temps):
        #ax.annotate(f"{T:.2f}", (S2[i], diff_coeffs[i]), fontsize=7)
    if len(S2) > 1:
        slope, intercept, r, p, se = stats.linregress(S2, np.log(diff_coeffs))
        r_squared = r**2
        line_fit = np.exp(slope*S2 + intercept)
        ax.plot(S2, line_fit, color='#FF9999', linewidth=2)
        if temp_change and density_change:
            ax.text(0.58, 0.95, f"D = exp({-slope:.3f} · -S₂/N + {intercept:.3f})\nR² = {r_squared:.4f}\nT={temp_start}-{temp_end}     ΔT={temp_step_size}\nρ={density_start}-{density_end}     Δρ={density_step_size}", 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif temp_change:
            ax.text(0.58, 0.95, f"D = exp({-slope:.3f} · -S₂/N + {intercept:.3f})\nR² = {r_squared:.4f}\nT={temp_start}-{temp_end}, ΔT={temp_step_size}\nρ={density_start}", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif density_change:
            ax.text(0.58, 0.95, f"D = exp({-slope:.3f} · -S₂/N + {intercept:.3f})\nR² = {r_squared:.4f}\nT={temp_start}\nρ= {density_start}-{density_end}, Δρ={density_step_size}", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel("-S2/N")
    ax.set_ylabel("D")
    ax.set_title(f"Diffusion Coefficient vs. Excess Entropy")
    plt.savefig(os.path.join(path, "plot.png"))
    plt.show()

    
if __name__ == "__main__":
    run_pipeline()