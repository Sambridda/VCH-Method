# materials/__init__.py
import os
import hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .fit import fit_material

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
XLSX_PATH = os.path.join(BASE_DIR, 'VCH DataSheet.xlsx')

def get_hash():
    if not os.path.exists(XLSX_PATH):
        raise FileNotFoundError(f"Could not find {XLSX_PATH}")
    with open(XLSX_PATH, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def _cache_valid(cache_file, current_hash):
    if not os.path.exists(cache_file): return False
    try:
        data = np.load(cache_file, allow_pickle=True)
        return str(data['hash']) == current_hash
    except:
        return False

def _plot_properties(name, data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    T = data['T_grid']
    axes[0, 0].plot(T, data['rho_grid'], 'b-'); axes[0, 0].set_title('Density (kg/m³)'); axes[0, 0].set_ylabel('kg/m³')
    axes[0, 1].plot(T, data['cp_sens_grid'], 'b-', label='Sensible')
    axes[0, 1].plot(T, data['cp_eff_grid'], 'r--', label='Effective')
    axes[0, 1].set_title('Cp (J/kgK)'); axes[0, 1].set_ylabel('J/kgK'); axes[0, 1].legend()
    axes[1, 0].plot(T, data['k_grid'], 'b-'); axes[1, 0].set_title('Conductivity (W/mK)'); axes[1, 0].set_ylabel('W/mK')
    axes[1, 1].semilogy(T, data['mu_grid'], 'b-'); axes[1, 1].set_title('Viscosity (Pa.s)'); axes[1, 1].set_ylabel('Pa.s')
    for ax in axes.flatten(): ax.grid(True, alpha=0.3)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f'{name}_properties.png'), dpi=100)
    plt.close(fig)

def check_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    current_hash = get_hash()
    xls = pd.ExcelFile(XLSX_PATH)
    
    for sheet in ['Fluids', 'Metals']:
        if sheet in xls.sheet_names:
            df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
            materials = set(df['Material'].dropna())
            for mat in materials:
                cache_file = os.path.join(CACHE_DIR, f'{mat}.npz')
                if _cache_valid(cache_file, current_hash): continue
                df_mat = df[df['Material'] == mat]
                try:
                    data = fit_material(df_mat)
                    np.savez(cache_file, hash=current_hash, **data)
                    _plot_properties(mat, data)
                except Exception as e:
                    print(f"Error fitting {mat}: {e}")

    if 'Insulation' in xls.sheet_names:
        df_insul = pd.read_excel(XLSX_PATH, sheet_name='Insulation')
        for _, row in df_insul.iterrows():
            mat = row['Material']
            if pd.isna(mat): continue
            cache_file = os.path.join(CACHE_DIR, f'{mat}.npz')
            if _cache_valid(cache_file, current_hash): continue
            k = float(row['k_WmK'])
            data = {
                'T_grid': np.array([0.0]), 'rho_grid': np.array([1000.0]),
                'cp_sens_grid': np.array([1000.0]), 'cp_eff_grid': np.array([1000.0]),
                'k_grid': np.array([k]), 'mu_grid': np.array([1e5]),
                'H_grid': np.array([0.0]), 'T_ms': np.array([np.nan]), 'T_me': np.array([np.nan])
            }
            np.savez(cache_file, hash=current_hash, **data)

def list_available():
    check_cache()
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npz')]
    return sorted([f[:-4] for f in files])

def load(name):
    cache_file = os.path.join(CACHE_DIR, f'{name}.npz')
    if not os.path.exists(cache_file):
        for f in os.listdir(CACHE_DIR):
            if f.lower() == f'{name.lower()}.npz':
                cache_file = os.path.join(CACHE_DIR, f)
                break
        else:
            raise FileNotFoundError(f"Material '{name}' not found in cache.")
            
    data = np.load(cache_file, allow_pickle=True)
    return {k: data[k] for k in data.files if k != 'hash'}

def load_settings():
    xls = pd.ExcelFile(XLSX_PATH)
    if 'Settings' in xls.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name='Settings')
        thresh_liq = float(df[df['Parameter'] == 'LiquidFraction_Threshold']['Value'].values[0])
        thresh_visc = float(df[df['Parameter'] == 'Viscosity_Threshold']['Value'].values[0])
        return thresh_liq, thresh_visc
    return 0.2, 5.0