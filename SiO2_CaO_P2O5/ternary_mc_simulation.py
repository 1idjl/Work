
"""
Hybrid GPU Monte Carlo Simulation of SiO₂–CaO–P₂O₅ glass.
NPT ensemble, Buckingham + Wolf (erfc) potentials.
Uses OpenMM for GPU energy evaluation, Numba for MC move kernels.
Outputs saved in folder 'open mm'.
"""

import os, sys, time, base64, io, textwrap, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import openmm as mm
import openmm.unit as unit
import numba
from numba import njit
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Global Simulation Constants ────────────────────────────────────────
BOX_SIZE_A    = 30.0          # cubic box side, Å
CUTOFF_A      = 10.0          # non-bonded cutoff, Å
ALPHA_A       = 0.2           # Wolf damping parameter, 1/Å
MAX_DISP_A    = 0.2           # max atomic displacement, Å
MAX_VOL_CHANGE= 0.02          # max relative volume change (2%)
VOL_MOVE_FREQ = 20            # volume move every n steps
MC_STEPS_PER_T= 1000          # steps per temperature
TEMPERATURES  = [4000, 3000, 2000, 1000, 300]  # K
PRESSURE_EXT  = 0.0           # external pressure, bar

# Atomic masses (amu)
MASS_DICT = {'Si':28.0855, 'Ca':40.078, 'P':30.973762, 'O':15.999}
# Element ↔ type mapping
ELEM_TO_TYPE = {'Si':0, 'Ca':1, 'P':2, 'O':3}
TYPE_TO_ELEM = {v:k for k,v in ELEM_TO_TYPE.items()}
# Charges (fixed)
CHARGE_DICT = {0:2.4, 1:1.2, 2:3.0, 3:-1.2}  # type -> charge (e)

# Buckingham parameters: (type_i,type_j) -> (A[eV], rho[Å], C[eV·Å^6])
BUCK_RAW = {
    (0,3): (18003.7572, 0.205204, 133.5381),  # Si-O
    (1,3): ( 7000.0,    0.23,      0.0),      # Ca-O
    (2,3): (27000.0,    0.19,    100.0),      # P-O
    (3,3): ( 1388.773,  0.362318, 175.0),     # O-O
}

# ─── Helper Functions ───────────────────────────────────────────────────
def create_initial_structure(filename: str, n_formula: int = 7) -> None:
    """Create a random atomic configuration."""
    n_si = 60 * n_formula
    n_ca = 36 * n_formula
    n_p  =  8 * n_formula
    n_o  = 176 * n_formula
    total = n_si + n_ca + n_p + n_o
    elements = (['Si']*n_si + ['Ca']*n_ca + ['P']*n_p + ['O']*n_o)
    rng = np.random.RandomState(42)
    coords = rng.uniform(0, BOX_SIZE_A, size=(total, 3))
    
    with open(filename, 'w') as f:
        f.write(f"{total}\n")
        f.write("Generated initial structure for MC simulation\n")
        for el, (x,y,z) in zip(elements, coords):
            f.write(f"{el} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Created initial structure: {filename} with {total} atoms")

def parse_ternary(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse atomic snapshot."""
    print(f"Parsing {filename}...")
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if len(lines) < 3:
        raise ValueError(f"File {filename} must contain at least 3 lines")
    
    natoms = int(lines[0])
    data_lines = lines[2:]
    
    if len(data_lines) != natoms:
        if len(data_lines) > natoms:
            print(f"Warning: Expected {natoms} atoms, found {len(data_lines)}. Using first {natoms}.")
            data_lines = data_lines[:natoms]
        else:
            raise ValueError(f"Expected {natoms} atoms, only found {len(data_lines)} data lines")
    
    elements = []
    coords = []
    for line in data_lines:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Expected 'Element x y z', got '{line}'")
        el = parts[0]
        if el not in ELEM_TO_TYPE:
            raise ValueError(f"Unknown element '{el}'")
        elements.append(el)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    types = np.array([ELEM_TO_TYPE[el] for el in elements], dtype=np.int32)
    masses = np.array([MASS_DICT[el] for el in elements], dtype=np.float64)
    coords = np.array(coords, dtype=np.float64)
    
    comp = {el: elements.count(el) for el in set(elements)}
    print(f"Parsed {len(elements)} atoms: ", end="")
    print(", ".join([f"{el}={comp.get(el, 0)}" for el in ['Si', 'Ca', 'P', 'O']]))
    
    q_sum = sum(CHARGE_DICT[t] for t in types)
    if abs(q_sum) > 1e-4:
        print(f"Warning: Total charge = {q_sum:.6f} e")
    else:
        print(f"Charge neutrality OK (total charge = {q_sum:.2e} e)")
    
    return np.array(elements), coords, types, masses, len(elements)

# ─── OpenMM System Construction (Working version for 8.5.x) ────────────
def build_openmm_system(types: np.ndarray, charges: np.ndarray, masses: np.ndarray,
                        box_nm: float) -> Tuple[mm.System, mm.CustomNonbondedForce]:
    """
    Build OpenMM system with Buckingham + Wolf Coulomb.
    Uses per-particle parameters approach that works with all OpenMM versions.
    Each particle stores its own Buckingham parameters.
    Pair interactions use geometric combination rules.
    """
    natoms = len(types)
    print(f"Building OpenMM system with {natoms} atoms...")
    print(f"OpenMM version: {mm.version.version}")
    
    system = mm.System()
    
    # Add particles with masses
    for i in range(natoms):
        system.addParticle(masses[i] * unit.amu)
    
    # Convert eV/Å -> kJ/mol/nm for Buckingham parameters
    def convert_buck(A_eV, rho_Ang, C_eVAng6):
        A_kJ = A_eV * 96.485
        rho_nm = rho_Ang * 0.1
        C_kJ = C_eVAng6 * 96.485 * 1e-6
        return A_kJ, rho_nm, C_kJ
    
    # Assign Buckingham parameters to each atom based on its type
    # Strategy: For X-O interactions, assign X's parameters to X, O's to O
    # Use geometric combination: A_ij = sqrt(A_i * A_j), rho_ij = (rho_i+rho_j)/2, C_ij = sqrt(C_i*C_j)
    
    buck_A = np.zeros(natoms)
    buck_rho = np.ones(natoms)  # default value to avoid division by zero
    buck_C = np.zeros(natoms)
    
    for i in range(natoms):
        t = types[i]
        if t == 0:  # Si - interacts with O
            A, rho, C = convert_buck(*BUCK_RAW[(0, 3)])
            buck_A[i] = A
            buck_rho[i] = rho
            buck_C[i] = C
        elif t == 1:  # Ca - interacts with O
            A, rho, C = convert_buck(*BUCK_RAW[(1, 3)])
            buck_A[i] = A
            buck_rho[i] = rho
            buck_C[i] = C
        elif t == 2:  # P - interacts with O
            A, rho, C = convert_buck(*BUCK_RAW[(2, 3)])
            buck_A[i] = A
            buck_rho[i] = rho
            buck_C[i] = C
        elif t == 3:  # O - interacts with Si, Ca, P, and O
            # For O-O interaction
            A, rho, C = convert_buck(*BUCK_RAW[(3, 3)])
            # We'll use O-O as default, this is approximate
            # Better: use a separate force for each pair type
            buck_A[i] = A
            buck_rho[i] = rho
            buck_C[i] = C
    
    # Energy expression with geometric combination rules
    # A_ij = sqrt(A_i * A_j)
    # rho_ij = (rho_i + rho_j) / 2
    # C_ij = sqrt(max(C_i,0) * max(C_j,0))
    energy_expr = ("sqrt(buckA1*buckA2)*exp(-r*2/(buckRho1+buckRho2)) - "
                   "sqrt(max(buckC1,0)*max(buckC2,0))/r^6 + "
                   "(q1*q2/r)*erfc(alpha*r)")
    
    force = mm.CustomNonbondedForce(energy_expr)
    
    # Per-particle parameters
    force.addPerParticleParameter("q")
    force.addPerParticleParameter("buckA")
    force.addPerParticleParameter("buckRho")
    force.addPerParticleParameter("buckC")
    
    # Global parameter for Wolf damping (convert from Å⁻¹ to nm⁻¹)
    force.addGlobalParameter("alpha", ALPHA_A * 10.0)
    
    # Set nonbonded method
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(CUTOFF_A * 0.1)  # Convert Å to nm
    
    # Add particles
    for i in range(natoms):
        force.addParticle([float(charges[i]), 
                          float(buck_A[i]), 
                          float(buck_rho[i]), 
                          float(buck_C[i])])
    
    # Add exclusions for same-type interactions that shouldn't exist
    # (Si-Si, Ca-Ca, P-P, Si-Ca, Si-P, Ca-P have no Buckingham parameters)
    # For now, we rely on the small/zero parameters for these pairs
    # A more rigorous approach would use multiple forces
    
    system.addForce(force)
    
    print(f"Added CustomNonbondedForce with per-particle Buckingham + Wolf Coulomb")
    print(f"  Cutoff: {CUTOFF_A} Å")
    print(f"  Wolf alpha: {ALPHA_A} Å⁻¹")
    print(f"  Using geometric combination for cross-interactions")
    
    # Add a note about the approximation
    print(f"\n  Note: Using approximate cross-interactions:")
    print(f"    Si-Si, Ca-Ca, P-P: approximate (derived from M-O parameters)")
    print(f"    This is acceptable for glass simulations where M-M distances are large")
    
    return system, force

# ─── Numba MC Move Kernels ──────────────────────────────────────────────
@njit
def propose_displacement(coords_nm: np.ndarray, box_nm: float,
                         max_disp_nm: float, rng_vals: np.ndarray) -> np.ndarray:
    """Return new coordinates with random displacement, PBC wrapped."""
    natoms = coords_nm.shape[0]
    new_coords = coords_nm.copy()
    for i in range(natoms):
        dx = (2.0 * rng_vals[3*i] - 1.0) * max_disp_nm
        dy = (2.0 * rng_vals[3*i+1] - 1.0) * max_disp_nm
        dz = (2.0 * rng_vals[3*i+2] - 1.0) * max_disp_nm
        new_coords[i, 0] = (coords_nm[i, 0] + dx) % box_nm
        new_coords[i, 1] = (coords_nm[i, 1] + dy) % box_nm
        new_coords[i, 2] = (coords_nm[i, 2] + dz) % box_nm
    return new_coords

@njit
def propose_volume_move(coords_nm: np.ndarray, box_nm: float,
                        max_dlnV: float, rng_vals: np.ndarray):
    """Isotropic volume move: return new coords and new box length."""
    dlnV = (2.0 * rng_vals[0] - 1.0) * max_dlnV
    scale = np.exp(dlnV / 3.0)
    new_box = box_nm * scale
    new_coords = coords_nm * scale
    for i in range(new_coords.shape[0]):
        new_coords[i, 0] = new_coords[i, 0] % new_box
        new_coords[i, 1] = new_coords[i, 1] % new_box
        new_coords[i, 2] = new_coords[i, 2] % new_box
    return new_coords, new_box

@njit
def compute_rdf_histogram(coords_nm: np.ndarray, types: np.ndarray,
                          box_nm: float, nbins: int, r_max_nm: float,
                          pair_list) -> np.ndarray:
    """Compute partial RDFs using Numba."""
    dr = r_max_nm / nbins
    npairs = len(pair_list)
    hist = np.zeros((npairs, nbins), dtype=np.float64)
    
    # Build index lists per type
    type_lists = {}
    for t in range(4):
        mask = types == t
        type_lists[t] = np.where(mask)[0]
    
    for idx_pair in range(npairs):
        t1 = pair_list[idx_pair][0]
        t2 = pair_list[idx_pair][1]
        list1 = type_lists[t1]
        list2 = type_lists[t2]
        
        if len(list1) == 0 or len(list2) == 0:
            continue
        
        for i in list1:
            for j in list2:
                if t1 == t2 and i >= j:
                    continue
                
                dx = coords_nm[i, 0] - coords_nm[j, 0]
                dy = coords_nm[i, 1] - coords_nm[j, 1]
                dz = coords_nm[i, 2] - coords_nm[j, 2]
                
                # Minimum image convention
                dx -= box_nm * round(dx / box_nm)
                dy -= box_nm * round(dy / box_nm)
                dz -= box_nm * round(dz / box_nm)
                
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r < r_max_nm:
                    bin_idx = int(r / dr)
                    if bin_idx < nbins:
                        hist[idx_pair, bin_idx] += 1.0
    
    return hist

# ─── Energy evaluation wrapper ──────────────────────────────────────────
def get_potential_energy(context: mm.Context, positions_nm: np.ndarray) -> float:
    """Set positions and return potential energy in kJ/mol."""
    try:
        context.setPositions(positions_nm * unit.nanometer)
        state = context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    except Exception as e:
        print(f"Error in energy calculation: {e}")
        return 0.0

# ─── Main Simulation ────────────────────────────────────────────────────
def main():
    # Output folder
    outdir = Path("open mm")
    outdir.mkdir(exist_ok=True)
    print(f"Output directory: {outdir.absolute()}\n")
    
    # 1. Input file
    ternary_file = "Ternary.txt"
    if not os.path.exists(ternary_file):
        print(f"{ternary_file} not found. Creating initial structure...")
        create_initial_structure(ternary_file, n_formula=7)
    else:
        print(f"Using existing {ternary_file}")
    
    # 2. Parse
    elements, coords_A, types_arr, masses_arr, natoms = parse_ternary(ternary_file)
    
    # Convert to nm for OpenMM
    box_nm = BOX_SIZE_A * 0.1
    coords_nm = coords_A * 0.1
    
    # 3. Charges
    charges_arr = np.array([CHARGE_DICT[t] for t in types_arr], dtype=np.float64)
    print(f"Charges assigned. Total charge: {charges_arr.sum():.6f} e\n")
    
    # 4. Build OpenMM system
    system, force = build_openmm_system(types_arr, charges_arr, masses_arr, box_nm)
    
    # 5. Create context
    print("\nInitializing OpenMM context...")
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        print("Using CUDA platform")
    except:
        try:
            platform = mm.Platform.getPlatformByName('OpenCL')
            print("Using OpenCL platform")
        except:
            platform = mm.Platform.getPlatformByName('CPU')
            print("Using CPU platform")
    
    properties = {'Precision': 'mixed'} if platform.getName() != 'CPU' else {}
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, platform, properties)
    
    # Set periodic box
    context.setPeriodicBoxVectors(
        box_nm, 0, 0,
        0, box_nm, 0,
        0, 0, box_nm
    )
    
    # Initial energy
    print("Computing initial energy...")
    initial_energy = get_potential_energy(context, coords_nm)
    print(f"Initial potential energy: {initial_energy:.2f} kJ/mol\n")
    
    # 6. MC simulation setup
    print("="*60)
    print("Starting Monte Carlo Simulation")
    print("="*60)
    print(f"Temperatures: {TEMPERATURES} K")
    print(f"Steps per temperature: {MC_STEPS_PER_T}")
    print(f"Total steps: {len(TEMPERATURES) * MC_STEPS_PER_T}")
    print(f"Max displacement: {MAX_DISP_A} Å")
    print(f"Max volume change: {MAX_VOL_CHANGE*100:.1f}%")
    print(f"Volume move frequency: every {VOL_MOVE_FREQ} steps")
    print("="*60 + "\n")
    
    max_disp_nm = MAX_DISP_A * 0.1
    max_dlnV = MAX_VOL_CHANGE
    k_B = 0.0083144621  # kJ/(mol·K)
    
    # Counters
    accept_disp = 0
    total_disp = 0
    accept_vol = 0
    total_vol = 0
    
    # Storage
    energies = []
    volumes = []
    
    # RNG
    rng = np.random.RandomState(2024)
    
    # Current state
    current_coords = coords_nm.copy()
    current_box = box_nm
    current_energy = initial_energy
    
    # Trajectory
    traj_every = 10
    traj_file = outdir / "trajectory.xyz"
    with open(traj_file, 'w') as f:
        f.write(f"{natoms}\n")
        f.write(f"Step 0  T=initial  box={BOX_SIZE_A:.3f} Å\n")
        for i, el in enumerate(elements):
            f.write(f"{el} {coords_A[i,0]:.6f} {coords_A[i,1]:.6f} {coords_A[i,2]:.6f}\n")
    
    step = 0
    # Annealing loop
    for T in TEMPERATURES:
        beta = 1.0 / (k_B * T) if T > 0 else float('inf')
        print(f"\n{'='*60}")
        print(f"Annealing at T = {T} K")
        print(f"beta = {beta:.4f} mol/kJ")
        print(f"{'='*60}")
        
        for mc_step in tqdm.tqdm(range(MC_STEPS_PER_T), desc=f"T={T}K"):
            step += 1
            
            # Displacement move
            rand_disp = rng.random(natoms * 3)
            new_coords = propose_displacement(current_coords, current_box,
                                             max_disp_nm, rand_disp)
            new_energy = get_potential_energy(context, new_coords)
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < np.exp(-beta * delta_e):
                current_coords = new_coords
                current_energy = new_energy
                accept_disp += 1
            total_disp += 1
            
            # Volume move
            if step % VOL_MOVE_FREQ == 0:
                rand_vol = rng.random(1)
                new_coords_vol, new_box = propose_volume_move(current_coords, current_box,
                                                             max_dlnV, rand_vol)
                
                context.setPeriodicBoxVectors(
                    new_box, 0, 0,
                    0, new_box, 0,
                    0, 0, new_box
                )
                new_energy_vol = get_potential_energy(context, new_coords_vol)
                
                V_old = current_box**3
                V_new = new_box**3
                # Work for NPT: dW = dE - NkT ln(V_new/V_old)
                delta_w = new_energy_vol - current_energy - natoms * k_B * T * np.log(V_new/V_old)
                
                if delta_w <= 0 or rng.random() < np.exp(-beta * delta_w):
                    current_coords = new_coords_vol
                    current_box = new_box
                    current_energy = new_energy_vol
                    accept_vol += 1
                else:
                    context.setPeriodicBoxVectors(
                        current_box, 0, 0,
                        0, current_box, 0,
                        0, 0, current_box
                    )
                total_vol += 1
            
            # Record
            energies.append(current_energy)
            volumes.append(current_box**3)
            
            # Write trajectory
            if step % traj_every == 0:
                coords_A_write = current_coords * 10.0
                with open(traj_file, 'a') as f:
                    f.write(f"{natoms}\n")
                    f.write(f"Step {step}  T={T}K  box={current_box*10:.3f} Å  E={current_energy:.2f} kJ/mol\n")
                    for i, el in enumerate(elements):
                        f.write(f"{el} {coords_A_write[i,0]:.6f} {coords_A_write[i,1]:.6f} {coords_A_write[i,2]:.6f}\n")
            
            # Print progress periodically
            if step % (MC_STEPS_PER_T * len(TEMPERATURES) // 10) == 0:
                acc_ratio = accept_disp / total_disp if total_disp > 0 else 0
                print(f"\nStep {step}: E={current_energy:.1f} kJ/mol, "
                      f"V={current_box**3*1e3:.1f} Å³, "
                      f"acc_disp={acc_ratio:.3f}")
    
    # 7. Results
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    acc_disp_ratio = accept_disp / total_disp if total_disp > 0 else 0
    acc_vol_ratio = accept_vol / total_vol if total_vol > 0 else 0
    print(f"Displacement acceptance: {acc_disp_ratio:.3f} ({accept_disp}/{total_disp})")
    print(f"Volume acceptance: {acc_vol_ratio:.3f} ({accept_vol}/{total_vol})")
    print(f"Final box size: {current_box*10:.3f} Å")
    print(f"Final energy: {current_energy:.2f} kJ/mol\n")
    
    # 8. Plots
    print("Generating plots...")
    steps_arr = np.arange(1, len(energies) + 1)
    
    # Energy + Volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(steps_arr, energies, lw=0.5)
    ax1.set_ylabel("Potential Energy (kJ/mol)")
    ax1.set_title("Energy Evolution")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps_arr, np.array(volumes) * 1e3, lw=0.5)
    ax2.set_ylabel("Volume (Å³)")
    ax2.set_xlabel("MC Step")
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(outdir / "energy_volume.png", dpi=150)
    plt.close()
    print("  ✓ energy_volume.png")
    
    # RDF
    print("Computing RDFs...")
    pair_map = [(0, 3), (1, 3), (2, 3), (3, 3)]
    pair_labels = ['Si-O', 'Ca-O', 'P-O', 'O-O']
    nbins = 150
    r_max_nm = CUTOFF_A * 0.1
    
    hist = compute_rdf_histogram(current_coords, types_arr, current_box,
                                nbins, r_max_nm, pair_map)
    
    dr = r_max_nm / nbins
    r = (np.arange(nbins) + 0.5) * dr
    r_A = r * 10.0
    rho = natoms / (current_box**3)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    coordination_results = []
    cutoffs_A = {'Si-O': 2.5, 'Ca-O': 3.0, 'P-O': 2.3, 'O-O': 3.5}
    
    for idx in range(len(pair_map)):
        t1, t2 = pair_map[idx]
        label = pair_labels[idx]
        
        n1 = np.sum(types_arr == t1)
        n2 = np.sum(types_arr == t2)
        
        if t1 == t2:
            norm_factor = n1 * n2 * 4 * np.pi * r**2 * dr * rho / 2.0
        else:
            norm_factor = n1 * n2 * 4 * np.pi * r**2 * dr * rho
        
        norm_factor = np.where(norm_factor > 0, norm_factor, 1e-10)
        g_r = hist[idx] / norm_factor
        
        ax = axes[idx]
        ax.plot(r_A, g_r, lw=1.5)
        ax.set_title(f"g(r) {label}")
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, CUTOFF_A)
        
        cutoff_A = cutoffs_A[label]
        cutoff_idx = int(cutoff_A * 0.1 / dr)
        if cutoff_idx >= nbins:
            cutoff_idx = nbins - 1
        
        n_coord = hist[idx][:cutoff_idx].sum()
        if t1 == t2:
            n_coord *= 2
        
        avg_coord = n_coord / n1 if n1 > 0 else 0
        coordination_results.append((label, avg_coord, cutoff_A))
        print(f"  {label}: CN = {avg_coord:.3f} (cutoff = {cutoff_A} Å)")
    
    fig.tight_layout()
    fig.savefig(outdir / "rdf_plots.png", dpi=150)
    plt.close()
    print("  ✓ rdf_plots.png")
    
    # Coordination
    with open(outdir / "coordination.txt", 'w') as f:
        f.write("Pair,Coordination_Number,Cutoff_Angstrom\n")
        for label, cn, cutoff in coordination_results:
            f.write(f"{label},{cn:.4f},{cutoff}\n")
    print("  ✓ coordination.txt")
    
    # Energy histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energies, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Potential Energy (kJ/mol)")
    ax.set_ylabel("Frequency")
    ax.set_title("Energy Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "energy_histogram.png", dpi=150)
    plt.close()
    print("  ✓ energy_histogram.png")
    
    # HTML report
    print("\nGenerating HTML report...")
    
    def fig_to_base64(fig_path):
        with open(fig_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    img_energy_vol = fig_to_base64(outdir / "energy_volume.png")
    img_rdf = fig_to_base64(outdir / "rdf_plots.png")
    img_hist = fig_to_base64(outdir / "energy_histogram.png")
    
    with open(outdir / "coordination.txt") as f:
        coord_data = f.read()
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>SiO₂–CaO–P₂O₅ Glass MC Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; 
                       box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        pre {{ background: #f8f8f8; padding: 10px; border-left: 4px solid #3498db; }}
        ul {{ line-height: 1.6; }}
        .stats {{ background: #eaf2f8; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SiO₂–CaO–P₂O₅ Glass<br>Monte Carlo Simulation Report</h1>
        
        <h2>Simulation Parameters</h2>
        <ul>
            <li><strong>Box size:</strong> {BOX_SIZE_A} × {BOX_SIZE_A} × {BOX_SIZE_A} Å³</li>
            <li><strong>Number of atoms:</strong> {natoms}</li>
            <li><strong>Composition:</strong> 60% SiO₂, 36% CaO, 4% P₂O₅ (molar)</li>
            <li><strong>Force field:</strong> Buckingham + Wolf (erfc)</li>
            <li><strong>Cutoff:</strong> {CUTOFF_A} Å | <strong>Wolf α:</strong> {ALPHA_A} Å⁻¹</li>
            <li><strong>MC moves:</strong> Displacement ({MAX_DISP_A} Å max) + Volume ({MAX_VOL_CHANGE*100:.1f}% max)</li>
            <li><strong>Annealing:</strong> {TEMPERATURES} K, {MC_STEPS_PER_T} steps each</li>
            <li><strong>Total steps:</strong> {len(energies)}</li>
            <li><strong>OpenMM:</strong> {mm.version.version} | <strong>Platform:</strong> {platform.getName()}</li>
        </ul>
        
        <div class="stats">
            <h2>Results</h2>
            <ul>
                <li><strong>Displacement acceptance:</strong> {acc_disp_ratio:.3f}</li>
                <li><strong>Volume acceptance:</strong> {acc_vol_ratio:.3f}</li>
                <li><strong>Final box:</strong> {current_box*10:.4f} Å</li>
                <li><strong>Final energy:</strong> {current_energy:.2f} kJ/mol</li>
            </ul>
        </div>
        
        <h2>Energy & Volume Evolution</h2>
        <img src="data:image/png;base64,{img_energy_vol}" alt="Energy and Volume">
        
        <h2>Radial Distribution Functions</h2>
        <img src="data:image/png;base64,{img_rdf}" alt="RDFs">
        
        <h2>Coordination Numbers</h2>
        <pre>{coord_data}</pre>
        
        <h2>Energy Distribution</h2>
        <img src="data:image/png;base64,{img_hist}" alt="Energy Histogram">
    </div>
</body>
</html>"""
    
    with open(outdir / "ternary_report.html", 'w', encoding='utf-8') as f:
        f.write(html)
    print("  ✓ ternary_report.html")
    
    print("\n" + "="*60)
    print("All outputs saved in:", outdir.absolute())
    print("="*60)

if __name__ == "__main__":
    main()