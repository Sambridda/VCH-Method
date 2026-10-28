# materials/fit.py
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def fit_material(df, T_min=-50.0, T_max=200.0, n_points=400):
    df = df.sort_values('Temperature_C').drop_duplicates('Temperature_C')
    T_data = df['Temperature_C'].values.astype(float)
    
    T_grid = np.linspace(T_min, T_max, n_points)
    
    def fit_col(col_name, default_val, log_space=False):
        if col_name not in df.columns:
            return np.full(n_points, default_val)
        y = df[col_name].values.astype(float)
        if log_space:
            y = np.log(np.maximum(y, 1e-6))
        f = PchipInterpolator(T_data, y)
        res = f(T_grid)
        if log_space:
            res = np.exp(res)
        return res

    rho_grid = fit_col('Density_kgm3', 1000.0)
    cp_sens_grid = fit_col('Cp_JkgK', 1000.0)
    k_grid = fit_col('k_WmK', 1.0)
    mu_grid = fit_col('Viscosity_Pas', 1e5, log_space=True)
    
    # Latent Heat
    if 'Latent_kJkg' in df.columns:
        latent_J = df['Latent_kJkg'].values.astype(float) * 1000.0
        f_lat = PchipInterpolator(T_data, latent_J)
        latent_grid = f_lat(T_grid)
    else:
        latent_grid = np.zeros(n_points)

    # Find Melt Band
    T_ms = np.nan
    T_me = np.nan
    if 'Phase' in df.columns:
        for idx, row in df.iterrows():
            if 'Melt Start' in str(row['Phase']): T_ms = row['Temperature_C']
            if 'End of Melt' in str(row['Phase']): T_me = row['Temperature_C']
            
    if not np.isnan(T_ms) and not np.isnan(T_me):
        latent_grid[(T_grid < T_ms) | (T_grid > T_me)] = 0.0
    else:
        latent_grid[:] = 0.0
        
    # Calculate effective heat capacity (sensible + phase change)
    dLdT = np.gradient(latent_grid, T_grid)
    cp_eff_grid = cp_sens_grid + dLdT
    
    # Clamp to prevent non-monotonic enthalpy (which crashes the solver)
    min_cp = 0.05 * cp_sens_grid
    cp_eff_grid = np.maximum(cp_eff_grid, min_cp)
    
    # Integrate H(T)
    H_grid = np.zeros(n_points)
    for i in range(1, n_points):
        cp_mid = 0.5 * (cp_eff_grid[i-1] + cp_eff_grid[i])
        H_grid[i] = H_grid[i-1] + cp_mid * (T_grid[i] - T_grid[i-1])
    H_grid -= H_grid[0]
    
    return {
        'T_grid': T_grid,
        'rho_grid': rho_grid,
        'cp_sens_grid': cp_sens_grid,
        'cp_eff_grid': cp_eff_grid,
        'k_grid': k_grid,
        'mu_grid': mu_grid,
        'H_grid': H_grid,
        'T_ms': T_ms,
        'T_me': T_me
    }