import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from tqdm import tqdm
import time

# تنظیمات نمایش
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
def read_pdb(filename):
    """خواندن فایل PDB و استخراج اطلاعات اتم‌ها"""
    atoms = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_type = line[76:78].strip()
                    if not atom_type:
                        atom_type = line[12:16].strip()[0]
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atoms.append({
                        'type': atom_type,
                        'x': x,
                        'y': y,
                        'z': z
                    })
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(atoms)
    print(f"✓ تعداد اتم‌های خوانده شده: {len(df)}")
    print(f"✓ انواع اتم‌ها: {df['type'].unique()}")
    return df
# پارامترهای Lennard-Jones (ε در kcal/mol، σ در Å)
LJ_PARAMS = {
    'Si': {'epsilon': 0.402, 'sigma': 3.826},
    'O': {'epsilon': 0.155, 'sigma': 3.166}
}

# بارهای الکتریکی (واحد: e)
CHARGES = {
    'Si': 2.4,
    'O': -1.2
}

# ثابت‌های فیزیکی
COULOMB_CONSTANT = 332.0636  # kcal·Å/(mol·e²)
BOLTZMANN = 0.001987204  # kcal/(mol·K)
def preprocess_for_numba(atom_types):
    """تبدیل atom_types به اندیس عددی و ساخت ماتریس پارامترها"""
    unique_types = list(set(atom_types))
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    atom_indices = np.array([type_to_idx[t] for t in atom_types], dtype=np.int32)
    
    n_types = len(unique_types)
    epsilon_matrix = np.zeros((n_types, n_types))
    sigma_matrix = np.zeros((n_types, n_types))
    charge_array = np.zeros(n_types)
    
    for i, type_i in enumerate(unique_types):
        charge_array[i] = CHARGES.get(type_i, 0.0)
        for j, type_j in enumerate(unique_types):
            eps_i = LJ_PARAMS.get(type_i, {}).get('epsilon', 0.0)
            eps_j = LJ_PARAMS.get(type_j, {}).get('epsilon', 0.0)
            sig_i = LJ_PARAMS.get(type_i, {}).get('sigma', 0.0)
            sig_j = LJ_PARAMS.get(type_j, {}).get('sigma', 0.0)
            
            epsilon_matrix[i, j] = np.sqrt(eps_i * eps_j)
            sigma_matrix[i, j] = (sig_i + sig_j) / 2.0
    
    return atom_indices, epsilon_matrix, sigma_matrix, charge_array
@jit(nopython=True)
def pbc_distance(r1, r2, box):
    """محاسبه فاصله با شرایط مرزی دوره‌ای"""
    delta = r1 - r2
    delta = delta - box * np.round(delta / box)
    return np.sqrt(np.sum(delta**2))

@jit(nopython=True)
def pbc_vector(r1, r2, box):
    """محاسبه بردار فاصله با PBC"""
    delta = r1 - r2
    delta = delta - box * np.round(delta / box)
    return delta
@jit(nopython=True)
def calculate_lj_energy_numba(positions, atom_indices, epsilon_matrix, sigma_matrix, box, cutoff):
    """محاسبه انرژی Lennard-Jones با Numba"""
    n_atoms = len(positions)
    energy = 0.0
    cutoff_sq = cutoff * cutoff
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = pbc_distance(positions[i], positions[j], box)
            
            if r < cutoff:
                type_i = atom_indices[i]
                type_j = atom_indices[j]
                
                epsilon = epsilon_matrix[type_i, type_j]
                sigma = sigma_matrix[type_i, type_j]
                
                if r > 0.1:  # جلوگیری از division by zero
                    sr6 = (sigma / r) ** 6
                    sr12 = sr6 * sr6
                    energy += 4.0 * epsilon * (sr12 - sr6)
    
    return energy
@jit(nopython=True)
def calculate_coulomb_energy_numba(positions, atom_indices, charge_array, box, cutoff, coulomb_const):
    """محاسبه انرژی Coulomb با Numba"""
    n_atoms = len(positions)
    energy = 0.0
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = pbc_distance(positions[i], positions[j], box)
            
            if r < cutoff:
                q_i = charge_array[atom_indices[i]]
                q_j = charge_array[atom_indices[j]]
                
                if r > 0.1:
                    energy += coulomb_const * q_i * q_j / r
    
    return energy
@jit(nopython=True)
def calculate_total_energy_numba(positions, atom_indices, epsilon_matrix, sigma_matrix, 
                                  charge_array, box, cutoff, coulomb_const):
    """محاسبه انرژی کل سیستم"""
    lj_energy = calculate_lj_energy_numba(positions, atom_indices, epsilon_matrix, 
                                           sigma_matrix, box, cutoff)
    coulomb_energy = calculate_coulomb_energy_numba(positions, atom_indices, charge_array, 
                                                     box, cutoff, coulomb_const)
    return lj_energy + coulomb_energy
@jit(nopython=True)
def random_move_atom(positions, atom_idx, max_displacement, box):
    """حرکت تصادفی یک اتم"""
    new_positions = positions.copy()
    displacement = (np.random.random(3) - 0.5) * 2 * max_displacement
    new_positions[atom_idx] += displacement
    new_positions[atom_idx] = new_positions[atom_idx] % box
    return new_positions
@jit(nopython=True)
def volume_move(positions, box, max_volume_change):
    """تغییر حجم جعبه شبیه‌سازی"""
    volume = box ** 3
    delta_v = (np.random.random() - 0.5) * 2 * max_volume_change
    new_volume = volume + delta_v
    
    if new_volume <= 0:
        return positions, box, False
    
    scale_factor = (new_volume / volume) ** (1.0 / 3.0)
    new_box = box * scale_factor
    new_positions = positions * scale_factor
    
    return new_positions, new_box, True
@jit(nopython=True)
def metropolis_criterion(delta_energy, temperature, boltzmann):
    """معیار پذیرش Metropolis"""
    if delta_energy < 0:
        return True
    else:
        probability = np.exp(-delta_energy / (boltzmann * temperature))
        return np.random.random() < probability
def npt_monte_carlo(df, n_steps=10000, temperature=300.0, pressure=1.0, 
                    max_displacement=0.1, max_volume_change=10.0, cutoff=10.0):
    """
    شبیه‌سازی مونت‌کارلو NPT با بهینه‌سازی Numba
    
    Parameters:
    -----------
    n_steps : int (پیش‌فرض: 10000)
    temperature : float (K)
    pressure : float (atm)
    cutoff : float (Å) - شعاع برش
    """
    
    print("\n" + "="*60)
    print("شروع شبیه‌سازی NPT Monte Carlo با Numba JIT")
    print("="*60)
    
    # آماده‌سازی داده‌ها
    positions = df[['x', 'y', 'z']].values.astype(np.float64)
    atom_types = df['type'].values
    n_atoms = len(positions)
    
    # محاسبه جعبه اولیه
    box_size = np.max([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]) * 1.2
    
    # پیش‌پردازش برای Numba
    atom_indices, epsilon_matrix, sigma_matrix, charge_array = preprocess_for_numba(atom_types)
    
    print(f"\n📊 پارامترهای شبیه‌سازی:")
    print(f"   • تعداد اتم‌ها: {n_atoms}")
    print(f"   • تعداد گام‌ها: {n_steps}")
    print(f"   • دما: {temperature} K")
    print(f"   • فشار: {pressure} atm")
    print(f"   • Cutoff: {cutoff} Å")
    print(f"   • اندازه جعبه اولیه: {box_size:.2f} Å")
    
    # محاسبه انرژی اولیه
    print("\n⏳ محاسبه انرژی اولیه...")
    current_energy = calculate_total_energy_numba(
        positions, atom_indices, epsilon_matrix, sigma_matrix,
        charge_array, box_size, cutoff, COULOMB_CONSTANT
    )
    print(f"✓ انرژی اولیه: {current_energy:.2f} kcal/mol")
    
    # آرایه‌های ذخیره نتایج
    energies = []
    volumes = []
    densities = []
    acceptance_atom = 0
    acceptance_volume = 0
    
    # شبیه‌سازی
    print(f"\n🚀 شروع {n_steps} گام شبیه‌سازی...\n")
    start_time = time.time()
    
    for step in tqdm(range(n_steps), desc="پیشرفت", ncols=80):
        # انتخاب نوع حرکت (90% اتم، 10% حجم)
        if np.random.random() < 0.9:
            # حرکت اتم
            atom_idx = np.random.randint(0, n_atoms)
            new_positions = random_move_atom(positions, atom_idx, max_displacement, box_size)
            new_box = box_size
            
            new_energy = calculate_total_energy_numba(
                new_positions, atom_indices, epsilon_matrix, sigma_matrix,
                charge_array, new_box, cutoff, COULOMB_CONSTANT
            )
            
            delta_energy = new_energy - current_energy
            
            if metropolis_criterion(delta_energy, temperature, BOLTZMANN):
                positions = new_positions
                current_energy = new_energy
                acceptance_atom += 1
        else:
            # تغییر حجم
            new_positions, new_box, valid = volume_move(positions, box_size, max_volume_change)
            
            if valid:
                new_energy = calculate_total_energy_numba(
                    new_positions, atom_indices, epsilon_matrix, sigma_matrix,
                    charge_array, new_box, cutoff, COULOMB_CONSTANT
                )
                
                volume = box_size ** 3
                new_volume = new_box ** 3
                delta_volume = new_volume - volume
                
                # اصلاح انرژی برای فشار (تبدیل atm به kcal/mol/Å³)
                pressure_term = pressure * 0.000101325 * delta_volume
                enthalpy_change = delta_energy + pressure_term - n_atoms * BOLTZMANN * temperature * np.log(new_volume / volume)
                
                if metropolis_criterion(enthalpy_change, temperature, BOLTZMANN):
                    positions = new_positions
                    box_size = new_box
                    current_energy = new_energy
                    acceptance_volume += 1
        
        # ذخیره نتایج
        energies.append(current_energy)
        volume = box_size ** 3
        volumes.append(volume)
        densities.append(n_atoms / volume)
    
    elapsed_time = time.time() - start_time
    
    # نتایج نهایی
    print(f"\n\n{'='*60}")
    print("✅ شبیه‌سازی با موفقیت به پایان رسید!")
    print(f"{'='*60}")
    print(f"\n⏱️  زمان اجرا: {elapsed_time:.2f} ثانیه ({elapsed_time/60:.2f} دقیقه)")
    print(f"\n📈 نرخ پذیرش:")
    print(f"   • حرکت اتم: {100*acceptance_atom/(0.9*n_steps):.1f}%")
    print(f"   • تغییر حجم: {100*acceptance_volume/(0.1*n_steps):.1f}%")
    print(f"\n📊 مقادیر نهایی:")
    print(f"   • انرژی: {energies[-1]:.2f} kcal/mol")
    print(f"   • حجم: {volumes[-1]:.2f} Å³")
    print(f"   • چگالی: {densities[-1]:.6f} atoms/Å³")
    
    results = {
        'energies': np.array(energies),
        'volumes': np.array(volumes),
        'densities': np.array(densities),
        'final_positions': positions,
        'final_box': box_size,
        'acceptance_atom': acceptance_atom,
        'acceptance_volume': acceptance_volume,
        'elapsed_time': elapsed_time
    }
    
    return results
def plot_results(results):
    """رسم نمودارهای نتایج"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # انرژی
    axes[0, 0].plot(results['energies'], linewidth=0.8, alpha=0.7)
    axes[0, 0].set_xlabel('گام شبیه‌سازی')
    axes[0, 0].set_ylabel('انرژی (kcal/mol)')
    axes[0, 0].set_title('تغییرات انرژی کل')
    axes[0, 0].grid(True, alpha=0.3)
    
    # حجم
    axes[0, 1].plot(results['volumes'], color='orange', linewidth=0.8, alpha=0.7)
    axes[0, 1].set_xlabel('گام شبیه‌سازی')
    axes[0, 1].set_ylabel('حجم (Å³)')
    axes[0, 1].set_title('تغییرات حجم')
    axes[0, 1].grid(True, alpha=0.3)
    
    # چگالی
    axes[1, 0].plot(results['densities'], color='green', linewidth=0.8, alpha=0.7)
    axes[1, 0].set_xlabel('گام شبیه‌سازی')
    axes[1, 0].set_ylabel('چگالی (atoms/Å³)')
    axes[1, 0].set_title('تغییرات چگالی')
    axes[1, 0].grid(True, alpha=0.3)
    
    # هیستوگرام انرژی
    axes[1, 1].hist(results['energies'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('انرژی (kcal/mol)')
    axes[1, 1].set_ylabel('فراوانی')
    axes[1, 1].set_title('توزیع انرژی')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('npt_simulation_results.png', dpi=300, bbox_inches='tight')
    print("\n💾 نمودارها در 'npt_simulation_results.png' ذخیره شدند")
    plt.show()


# ==================== اجرای اصلی ====================
if __name__ == "__main__":
    # خواندن فایل PDB
    pdb_file = "SiO2_21A_3d.pdb"  # نام فایل خود را اینجا بنویسید
    df = read_pdb(pdb_file)
    
    # اجرای شبیه‌سازی
    results = npt_monte_carlo(
        df,
        n_steps=10000,        # 10000 گام
        temperature=300.0,    # 300 کلوین
        pressure=1.0,         # 1 اتمسفر
        cutoff=10.0           # 10 انگستروم
    )
    
    # رسم نمودارها
    plot_results(results)
    
    print("\n✅ تمام مراحل با موفقیت انجام شد!")
