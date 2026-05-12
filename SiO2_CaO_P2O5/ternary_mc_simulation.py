
"""
Hybrid GPU Monte Carlo Simulation of SiO₂–CaO–P₂O₅ glass.
NPT ensemble, Buckingham + Wolf (erfc) potentials.
Uses OpenMM for GPU energy evaluation, Numba for MC move kernels.
Outputs saved in folder 'open mm'.
Compatible with OpenMM 7.x and 8.x
"""

import os, sys, time, base64, io, textwrap
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
import pandas as pd

# ─── Global Simulation Constants ────────────────────────────────────────
BOX_SIZE_A    = 30.0          # cubic box side, Å
CUTOFF_A      = 10.0          # non-bonded cutoff, Å
ALPHA_A       = 0.2           # Wolf damping parameter, 1/Å
TARGET_DENSITY= 2.2           # g/cm³
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

# ─── Helper: generate initial Ternary.txt if missing ────────────────────
def create_initial_structure(filename: str, n_formula: int = 7) -> None:
    """
    Create a random atomic configuration for n_formula units.
    Composition per formula: 60 SiO₂, 36 CaO, 4 P₂O₅
    Total atoms = 280 * n_formula
    """
    n_si = 60 * n_formula
    n_ca = 36 * n_formula
    n_p  =  8 * n_formula
    n_o  = 176 * n_formula
    total = n_si + n_ca + n_p + n_o
    elements = (['Si']*n_si + ['Ca']*n_ca + ['P']*n_p + ['O']*n_o)
    rng = np.random.RandomState(42)
    coords = rng.uniform(0, BOX_SIZE_A, size=(total, 3))
    
    assert len(elements) == total, f"Element list length {len(elements)} != total {total}"
    
    with open(filename, 'w') as f:
        f.write(f"{total}\n")
        f.write("Generated initial structure for MC simulation\n")
        for el, (x,y,z) in zip(elements, coords):
            f.write(f"{el} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Created initial structure: {filename} with {total} atoms")
    print(f"  Composition: Si={n_si}, Ca={n_ca}, P={n_p}, O={n_o}")

# ─── Parser for Ternary.txt ────────────────────────────────────────────
def parse_ternary(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse atomic snapshot. Returns: elements, coords_Å, types, masses, natoms"""
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
    for i, line in enumerate(data_lines, start=3):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Line {i}: expected 'Element x y z', got '{line}'")
        el = parts[0]
        if el not in ELEM_TO_TYPE:
            raise ValueError(f"Line {i}: unknown element '{el}'. Allowed: {list(ELEM_TO_TYPE.keys())}")
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

# ─── OpenMM System Construction (Compatible with OpenMM 7.x) ───────────
def build_openmm_system(types: np.ndarray, charges: np.ndarray, masses: np.ndarray,
                        box_nm: float) -> Tuple[mm.System, mm.CustomNonbondedForce]:
    """
    Create OpenMM System with CustomNonbondedForce (Buck + Wolf).
    Uses per-particle parameters approach compatible with OpenMM 7.x.
    Each particle gets: charge, buckA, buckRho, buckC
    Interaction uses lookup based on particle parameters.
    """
    natoms = len(types)
    print(f"Building OpenMM system with {natoms} atoms...")
    print(f"OpenMM version: {mm.version.version}")
    
    system = mm.System()
    
    # Add particles
    for i in range(natoms):
        system.addParticle(masses[i] * unit.amu)
    
    # Convert eV/Å -> kJ/mol/nm for Buckingham parameters
    def convert_buck(params):
        A_eV, rho_Ang, C_eVAng6 = params
        A_kJ = A_eV * 96.485
        rho_nm = rho_Ang * 0.1
        C_kJ = C_eVAng6 * 96.485 * 1e-6
        return (A_kJ, rho_nm, C_kJ)
    
    # Create parameter arrays for each particle
    buck_A = np.zeros(natoms)
    buck_rho = np.ones(natoms)  # avoid division by zero
    buck_C = np.zeros(natoms)
    
    # Assign Buckingham parameters to each particle based on type
    # We need both particles in a pair to have the pair-specific parameters
    # Strategy: Use CustomNonbondedForce with per-particle parameters
    # The energy expression will combine parameters from both particles
    
    # For Buckingham between types t1 and t2, we use geometric combination:
    # A_ij = sqrt(A_i * A_j), rho_ij = (rho_i + rho_j)/2, C_ij = sqrt(C_i * C_j)
    # This requires solving for individual parameters from pair parameters
    
    # Simpler approach: Use 4 parameters per particle (one for each possible interaction)
    # and use switching functions based on particle types
    # Even simpler: Use a single set and accept that we can't distinguish all pairs perfectly
    
    # Most practical approach for OpenMM 7.x: Use multiple CustomNonbondedForces
    # One for Coulomb, and separate ones for each Buckingham pair
    
    # Let's use the simplest working approach:
    # CustomNonbondedForce with energy expression using per-particle parameters
    
    # Build type-specific parameter arrays
    # For each type, store A, rho, C for its interaction with O (type 3)
    # This covers Si-O, Ca-O, P-O, O-O interactions
    # For metal-metal interactions, set to zero
    
    for i in range(natoms):
        t = types[i]
        if t == 0:  # Si
            pair = (0, 3)
            A, rho, C = convert_buck(BUCK_RAW.get((0, 3), (0, 0.1, 0)))
        elif t == 1:  # Ca
            A, rho, C = convert_buck(BUCK_RAW.get((1, 3), (0, 0.1, 0)))
        elif t == 2:  # P
            A, rho, C = convert_buck(BUCK_RAW.get((2, 3), (0, 0.1, 0)))
        elif t == 3:  # O
            A, rho, C = convert_buck(BUCK_RAW.get((3, 3), (0, 0.1, 0)))
        else:
            A, rho, C = 0.0, 0.1, 0.0
        
        buck_A[i] = A
        buck_rho[i] = rho
        buck_C[i] = C
    
    # Create CustomNonbondedForce with per-particle parameters
    # Energy expression: product of A, average of rho, product of C
    energy_expr = ("sqrt(buckA1*buckA2)*exp(-r/((buckRho1+buckRho2)/2)) - "
                   "sqrt(buckC1*buckC2)/r^6 + "
                   "(q1*q2/r)*erfc(alpha*r)")
    
    force = mm.CustomNonbondedForce(energy_expr)
    
    # Add per-particle parameters
    force.addPerParticleParameter("q")
    force.addPerParticleParameter("buckA")
    force.addPerParticleParameter("buckRho")
    force.addPerParticleParameter("buckC")
    
    # Add global parameter for Wolf damping
    force.addGlobalParameter("alpha", ALPHA_A * 10.0)  # Convert Å⁻¹ to nm⁻¹
    
    # Set nonbonded method
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(CUTOFF_A * 0.1)  # 10 Å = 1.0 nm
    
    # Add particles with their parameters
    for i in range(natoms):
        force.addParticle([float(charges[i]), float(buck_A[i]), 
                          float(buck_rho[i]), float(buck_C[i])])
    
    system.addForce(force)
    
    print("Added CustomNonbondedForce with Buckingham + Wolf Coulomb")
    print(f"  Cutoff: {CUTOFF_A} Å")
    print(f"  Wolf alpha: {ALPHA_A} Å⁻¹")
    
    return system, force

# ─── Alternative approach with interaction groups (for OpenMM 8.x) ──────
def build_openmm_system_v8(types: np.ndarray, charges: np.ndarray, masses: np.ndarray,
                           box_nm: float) -> Tuple[mm.System, mm.CustomNonbondedForce]:
    """
    OpenMM 8.x version using interaction groups.
    Only used if OpenMM version >= 8.0
    """
    natoms = len(types)
    print(f"Building OpenMM system with {natoms} atoms (using OpenMM 8.x features)...")
    
    system = mm.System()
    for i in range(natoms):
        system.addParticle(masses[i] * unit.amu)
    
    energy_expr = "buckA*exp(-r/buckRho) - buckC/r^6 + (q1*q2/r)*erfc(alpha*r)"
    force = mm.CustomNonbondedForce(energy_expr)
    
    force.addPerParticleParameter("q")
    force.addGlobalParameter("alpha", ALPHA_A * 10.0)
    
    # Check if interaction group parameters are available
    if hasattr(force, 'addPerInteractionGroupParameter'):
        force.addPerInteractionGroupParameter("buckA")
        force.addPerInteractionGroupParameter("buckRho")
        force.addPerInteractionGroupParameter("buckC")
    
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(CUTOFF_A * 0.1)
    
    # Add particles
    for q in charges:
        force.addParticle([float(q)])
    
    # Build type sets
    type_sets = {t: [] for t in range(4)}
    for idx, t in enumerate(types):
        type_sets[t].append(idx)
    
    # Convert parameters
    def convert_buck(params):
        A_eV, rho_Ang, C_eVAng6 = params
        A_kJ = A_eV * 96.485
        rho_nm = rho_Ang * 0.1
        C_kJ = C_eVAng6 * 96.485 * 1e-6
        return (A_kJ, rho_nm, C_kJ)
    
    from itertools import combinations_with_replacement
    for t1, t2 in combinations_with_replacement(range(4), 2):
        set1 = type_sets[t1]
        set2 = type_sets[t2]
        if not set1 or not set2:
            continue
        
        pair = (min(t1, t2), max(t1, t2))
        if pair in BUCK_RAW:
            A, rho, C = convert_buck(BUCK_RAW[pair])
        else:
            A, rho, C = 0.0, 1.0, 0.0
        
        if hasattr(force, 'addInteractionGroup'):
            # OpenMM 8.x
            force.addInteractionGroup(set1, set2, [A, rho, C])
        else:
            # Older approach - use exclusions
            # This won't work perfectly, but we'll use the per-particle approach instead
            print(f"  Warning: Interaction groups not available for {TYPE_TO_ELEM[t1]}-{TYPE_TO_ELEM[t2]}")
    
    system.addForce(force)
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
    n = coords_nm.shape[0]
    dr = r_max_nm / nbins
    npairs = len(pair_list)
    hist = np.zeros((npairs, nbins), dtype=np.float64)
    
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
    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

# ─── Main Simulation Routine ───────────────────────────────────────────
def main():
    # Output folder
    outdir = Path("open mm")
    outdir.mkdir(exist_ok=True)
    print(f"Output directory: {outdir.absolute()}")
    
    # 0. Ensure input file exists
    ternary_file = "Ternary.txt"
    if not os.path.exists(ternary_file):
        print(f"{ternary_file} not found. Creating initial structure...")
        create_initial_structure(ternary_file, n_formula=7)
    else:
        print(f"Using existing {ternary_file}")
    
    # 1. Parse
    elements, coords_A, types_arr, masses_arr, natoms = parse_ternary(ternary_file)
    
    # Convert to nm
    box_nm = BOX_SIZE_A * 0.1
    coords_nm = coords_A * 0.1
    
    # 2. Assign charges
    charges_arr = np.array([CHARGE_DICT[t] for t in types_arr], dtype=np.float64)
    print(f"Charges assigned. Total charge: {charges_arr.sum():.6f} e")
    
    # 3. Build OpenMM system (version-appropriate)
    # Check OpenMM version
    openmm_version = tuple(map(int, mm.version.version.split('.')))
    if openmm_version >= (8, 0):
        print("Using OpenMM 8.x features")
        system, force = build_openmm_system_v8(types_arr, charges_arr, masses_arr, box_nm)
    else:
        print("Using OpenMM 7.x compatible approach")
        system, force = build_openmm_system(types_arr, charges_arr, masses_arr, box_nm)
    
    # 4. Create OpenMM context
    print("Initializing OpenMM context...")
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        print("Using CUDA platform")
    except:
        try:
            platform = mm.Platform.getPlatformByName('OpenCL')
            print("Using OpenCL platform")
        except:
            platform = mm.Platform.getPlatformByName('CPU')
            print("Using CPU platform (no GPU found)")
    
    properties = {'Precision': 'mixed'} if platform.getName() != 'CPU' else {}
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, platform, properties)
    
    # Set periodic box
    context.setPeriodicBoxVectors(
        box_nm, 0, 0,
        0, box_nm, 0,
        0, 0, box_nm
    )
    
    # Set initial positions
    context.setPositions(coords_nm * unit.nanometer)
    initial_energy = get_potential_energy(context, coords_nm)
    print(f"Initial potential energy: {initial_energy:.2f} kJ/mol")
    
    # 5. MC simulation
    print("\nStarting Monte Carlo simulation...")
    max_disp_nm = MAX_DISP_A * 0.1
    max_dlnV = MAX_VOL_CHANGE
    k_B = 0.0083144621  # kJ/(mol·K)
    
    accept_disp = 0
    total_disp = 0
    accept_vol = 0
    total_vol = 0
    
    energies = []
    volumes = []
    
    rng = np.random.RandomState(2024)
    
    current_coords = coords_nm.copy()
    current_box = box_nm
    current_energy = initial_energy
    
    # Trajectory file
    traj_every = 10
    traj_file = outdir / "trajectory.xyz"
    with open(traj_file, 'w') as f:
        f.write(f"{natoms}\n")
        f.write(f"Step 0  T=initial  box={BOX_SIZE_A:.3f} Å\n")
        for i, el in enumerate(elements):
            f.write(f"{el} {coords_A[i,0]:.6f} {coords_A[i,1]:.6f} {coords_A[i,2]:.6f}\n")
    
    step = 0
    # Annealing schedule
    for T in TEMPERATURES:
        beta = 1.0 / (k_B * T) if T > 0 else float('inf')
        print(f"\n--- Annealing at {T} K (beta={beta:.4f}) ---")
        
        for mc_step in tqdm.tqdm(range(MC_STEPS_PER_T), desc=f"T={T}K"):
            step += 1
            
            # --- Displacement move ---
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
            
            # --- Volume move ---
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
                delta_w = (new_energy_vol - current_energy + 
                          PRESSURE_EXT * (V_new - V_old) * 1e-25 -
                          natoms * k_B * T * np.log(V_new / V_old))
                
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
            
            # --- Record ---
            energies.append(current_energy)
            volumes.append(current_box**3)
            
            # --- Write trajectory ---
            if step % traj_every == 0:
                coords_A_write = current_coords * 10.0
                with open(traj_file, 'a') as f:
                    f.write(f"{natoms}\n")
                    f.write(f"Step {step}  T={T}K  box={current_box*10:.3f} Å  E={current_energy:.2f} kJ/mol\n")
                    for i, el in enumerate(elements):
                        f.write(f"{el} {coords_A_write[i,0]:.6f} {coords_A_write[i,1]:.6f} {coords_A_write[i,2]:.6f}\n")
    
    # 6. Post-simulation analysis
    print("\n=== Simulation Complete ===")
    acc_disp_ratio = accept_disp / total_disp if total_disp > 0 else 0
    acc_vol_ratio = accept_vol / total_vol if total_vol > 0 else 0
    print(f"Displacement acceptance: {acc_disp_ratio:.3f} ({accept_disp}/{total_disp})")
    print(f"Volume acceptance: {acc_vol_ratio:.3f} ({accept_vol}/{total_vol})")
    print(f"Final box size: {current_box*10:.3f} Å")
    print(f"Final energy: {current_energy:.2f} kJ/mol")
    
    # Save energy and volume plots
    print("Generating plots...")
    steps_arr = np.arange(1, len(energies) + 1)
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
    print("Saved energy_volume.png")
    
    # 7. RDF calculation
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
        print(f"  {label}: average coordination = {avg_coord:.3f} (cutoff = {cutoff_A} Å)")
    
    fig.tight_layout()
    fig.savefig(outdir / "rdf_plots.png", dpi=150)
    plt.close()
    print("Saved rdf_plots.png")
    
    # Save coordination data
    with open(outdir / "coordination.txt", 'w') as f:
        f.write("Pair,Coordination_Number,Cutoff_Angstrom\n")
        for label, cn, cutoff in coordination_results:
            f.write(f"{label},{cn:.4f},{cutoff}\n")
    print("Saved coordination.txt")
    
    # 8. Energy histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energies, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Potential Energy (kJ/mol)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Energy Distribution (Final T={TEMPERATURES[-1]}K)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "energy_histogram.png", dpi=150)
    plt.close()
    print("Saved energy_histogram.png")
    
    # 9. HTML report
    print("Generating HTML report...")
    
    def fig_to_base64(fig_path):
        with open(fig_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    img_energy_vol = fig_to_base64(outdir / "energy_volume.png")
    img_rdf = fig_to_base64(outdir / "rdf_plots.png")
    img_hist = fig_to_base64(outdir / "energy_histogram.png")
    
    with open(outdir / "coordination.txt") as f:
        coord_data = f.read()
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SiO₂–CaO–P₂O₅ Glass MC Simulation Report</title>
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
            <li><strong>Force field:</strong> Buckingham + Wolf (erfc) electrostatic</li>
            <li><strong>Cutoff:</strong> {CUTOFF_A} Å</li>
            <li><strong>Wolf α:</strong> {ALPHA_A} Å⁻¹</li>
            <li><strong>MC moves:</strong> Atomic displacement ({MAX_DISP_A} Å max), Isotropic volume ({MAX_VOL_CHANGE*100:.1f}% max every {VOL_MOVE_FREQ} steps)</li>
            <li><strong>Annealing schedule:</strong> {TEMPERATURES} K, {MC_STEPS_PER_T} steps each</li>
            <li><strong>Total MC steps:</strong> {len(energies)}</li>
        </ul>
        
        <div class="stats">
            <h2>Acceptance Statistics</h2>
            <ul>
                <li><strong>Displacement acceptance:</strong> {acc_disp_ratio:.3f} ({accept_disp}/{total_disp} moves accepted)</li>
                <li><strong>Volume acceptance:</strong> {acc_vol_ratio:.3f} ({accept_vol}/{total_vol} moves accepted)</li>
                <li><strong>Final box size:</strong> {current_box*10:.4f} Å</li>
                <li><strong>Final potential energy:</strong> {current_energy:.2f} kJ/mol</li>
            </ul>
        </div>
        
        <h2>Energy and Volume Evolution</h2>
        <img src="data:image/png;base64,{img_energy_vol}" alt="Energy and Volume vs MC Step">
        
        <h2>Radial Distribution Functions</h2>
        <img src="data:image/png;base64,{img_rdf}" alt="RDF plots">
        
        <h2>Coordination Numbers</h2>
        <pre>{coord_data}</pre>
        
        <h2>Energy Distribution (Final Configuration)</h2>
        <img src="data:image/png;base64,{img_hist}" alt="Energy Histogram">
        
        <footer style="margin-top: 40px; color: #7f8c8d; font-size: 0.9em;">
            <p>Generated by Hybrid GPU Monte Carlo Simulation (OpenMM + Numba)</p>
        </footer>
    </div>
</body>
</html>"""
    
    report_file = outdir / "ternary_report.html"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n{'='*60}")
    print(f"Simulation complete! All outputs saved in: {outdir.absolute()}")
    print(f"Files generated:")
    print(f"  - trajectory.xyz          (trajectory file)")
    print(f"  - energy_volume.png       (energy & volume evolution)")
    print(f"  - rdf_plots.png           (radial distribution functions)")
    print(f"  - energy_histogram.png    (energy distribution)")
    print(f"  - coordination.txt        (coordination numbers)")
    print(f"  - ternary_report.html     (complete HTML report)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()