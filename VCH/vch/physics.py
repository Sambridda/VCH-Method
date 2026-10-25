"""Physics: material tables, enthalpy <-> temperature, oil / water properties,
internal and external film coefficients and the epsilon-NTU coil duty.

Everything here is a direct port of P6.  The heavy functions are Numba-safe
(`njit`) so the sweep kernels in sweep.py / cooldown.py can call them.
"""
import numpy as np
from numba import njit

# Cached + GIL-free so Phase A can run several designs in parallel threads.
_jit = njit(cache=True, nogil=True)

# ------------------------------------------------------------
# WATER REFERENCE TABLE (10-100 degC)
# ------------------------------------------------------------
WATER_T = np.array([10., 20., 30., 40., 50., 60., 70., 80., 90., 100.])
WATER_RHO = np.array([999.7, 998.2, 995.7, 992.2, 988.0, 983.2, 977.8, 971.8, 965.3, 958.4])
WATER_CP = np.array([4192., 4182., 4178., 4179., 4181., 4184., 4187., 4197., 4205., 4216.])
WATER_MU = np.array([1.308e-3, 1.002e-3, 7.98e-4, 6.53e-4, 5.47e-4,
                     4.66e-4, 4.04e-4, 3.55e-4, 3.15e-4, 2.82e-4])
WATER_K = np.array([0.580, 0.598, 0.615, 0.631, 0.644, 0.654, 0.663, 0.670, 0.675, 0.679])


# ------------------------------------------------------------
# MATERIAL TABLES (enthalpy method)
# ------------------------------------------------------------
def build_material_tables(cfg):
    T_ms = cfg["T_ms"]; T_me = cfg["T_me"]; Hfus = cfg["Hfus"]
    T_amb_low = cfg["T_amb_low"]; T_water_in = cfg["T_water_in"]

    T_min = min(T_amb_low, T_ms) - 20.0
    T_max = max(T_water_in, T_me) + 20.0
    n_g = 400
    T_grid = np.linspace(T_min, T_max, n_g)

    Ts, Tl = cfg["T_solid_arr"], cfg["T_liquid_arr"]
    rho_s_arr, cp_s_arr, k_s_arr, mu_s_arr = (cfg["rho_solid_arr"], cfg["cp_solid_arr"],
                                              cfg["k_solid_arr"], cfg["mu_solid_arr"])
    rho_l_arr, cp_l_arr, k_l_arr, mu_l_arr = (cfg["rho_liquid_arr"], cfg["cp_liquid_arr"],
                                              cfg["k_liquid_arr"], cfg["mu_liquid_arr"])

    dT_melt = T_me - T_ms
    Hfus_over_dT = Hfus / dT_melt

    rho_grid = np.empty(n_g); cp_sens_grid = np.empty(n_g)
    cp_eff_grid = np.empty(n_g); k_grid = np.empty(n_g); mu_grid = np.empty(n_g)

    rho_s_end = float(np.interp(T_ms, Ts, rho_s_arr))
    rho_l_start = float(np.interp(T_me, Tl, rho_l_arr))
    cp_s_end = float(np.interp(T_ms, Ts, cp_s_arr))
    cp_l_start = float(np.interp(T_me, Tl, cp_l_arr))
    k_s_end = float(np.interp(T_ms, Ts, k_s_arr))
    k_l_start = float(np.interp(T_me, Tl, k_l_arr))
    mu_l_start = float(np.interp(T_me, Tl, mu_l_arr))

    for i in range(n_g):
        T = T_grid[i]
        if T < T_ms:
            rho_grid[i] = np.interp(T, Ts, rho_s_arr)
            cp_sens_grid[i] = np.interp(T, Ts, cp_s_arr)
            k_grid[i] = np.interp(T, Ts, k_s_arr)
            mu_grid[i] = np.interp(T, Ts, mu_s_arr)
            cp_eff_grid[i] = cp_sens_grid[i]
        elif T > T_me:
            rho_grid[i] = np.interp(T, Tl, rho_l_arr)
            cp_sens_grid[i] = np.interp(T, Tl, cp_l_arr)
            k_grid[i] = np.interp(T, Tl, k_l_arr)
            mu_grid[i] = np.interp(T, Tl, mu_l_arr)
            cp_eff_grid[i] = cp_sens_grid[i]
        else:
            f = (T - T_ms) / dT_melt
            rho_grid[i] = rho_s_end * (1 - f) + rho_l_start * f
            cp_sens_grid[i] = cp_s_end * (1 - f) + cp_l_start * f
            k_grid[i] = k_s_end * (1 - f) + k_l_start * f
            mu_grid[i] = mu_l_start  # liquid viscosity for slurry flow
            cp_eff_grid[i] = cp_sens_grid[i] + Hfus_over_dT

    H_grid = np.zeros(n_g)
    for i in range(1, n_g):
        H_grid[i] = H_grid[i - 1] + 0.5 * (cp_eff_grid[i - 1] + cp_eff_grid[i]) * (T_grid[i] - T_grid[i - 1])

    return dict(T_grid=T_grid, H_grid=H_grid,
                rho_grid=rho_grid, cp_sens_grid=cp_sens_grid, k_grid=k_grid, mu_grid=mu_grid)


# ------------------------------------------------------------
# HEAT TRANSFER
# ------------------------------------------------------------
@_jit
def h_internal(m_dot, d_in, d_coil_actual, rho, cp, mu, k):
    A = np.pi * (d_in / 2.0) ** 2
    v = m_dot / (rho * A)
    Re = rho * v * d_in / mu
    Pr = mu * cp / k
    if Re > 10000.0:
        f = (0.79 * np.log(Re) - 1.64) ** -2
        Nu = (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 2.17 * np.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
        Nu *= (1.0 + (d_in / d_coil_actual) ** 0.9)  # actual helix diameter
    else:
        Nu = 4.36
    return Nu * k / d_in


@_jit
def h_external(T_oil, T_grid, rho_grid, cp_grid, k_grid, mu_grid, T_ms,
               has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural, d_out):
    if T_oil < T_ms and has_agitator == 0:
        return 5.0  # unagitated solid

    rho = np.interp(T_oil, T_grid, rho_grid)
    cp = np.interp(T_oil, T_grid, cp_grid)
    k = np.interp(T_oil, T_grid, k_grid)
    mu = np.interp(T_oil, T_grid, mu_grid)

    if has_agitator == 1:
        N_rps = N_rpm / 60.0
        Re_ia = rho * N_rps * (D_imp ** 2) / mu
        Pr = mu * cp / k
        if ag_type == 1:
            C, a = 0.60, 0.67
        elif ag_type == 2:
            C, a = 0.54, 0.67
        elif ag_type == 3:
            C, a = 0.40, 0.50
        else:
            C, a = 0.35, 0.50
        Nu = C * (Re_ia ** a) * (Pr ** (1.0 / 3.0)) * visc_corr
        return Nu * k / d_out
    else:
        nu = mu / rho
        alpha = k / (rho * cp)
        Pr = nu / alpha
        d_char = d_out  # tube OD for natural convection
        Ra = 9.81 * beta * dT_natural * (d_char ** 3) / (nu * alpha)
        Nu = (0.60 + (0.387 * Ra ** (1.0 / 6.0)) / (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
        return Nu * k / d_char


@_jit
def calc_Q(m_dot, T_band, T_water_in, d_in, d_out, R_f_out, A_coil, d_coil_actual,
           T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
           T_tube_arr, k_tube_arr,
           T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms,
           has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural):
    """Coil duty (W) by the epsilon-NTU method."""
    if T_band >= T_water_in:
        return 0.0
    T_eff = 0.5 * (T_water_in + T_band)

    rho_c = np.interp(T_eff, T_water_arr, rho_water_arr)
    cp_c = np.interp(T_eff, T_water_arr, cp_water_arr)
    mu_c = np.interp(T_eff, T_water_arr, mu_water_arr)
    k_c = np.interp(T_eff, T_water_arr, k_water_arr)
    k_tube = np.interp(T_eff, T_tube_arr, k_tube_arr)

    h_in = h_internal(m_dot, d_in, d_coil_actual, rho_c, cp_c, mu_c, k_c)
    h_out = h_external(T_band, T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms,
                       has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural, d_out)

    R_in = 1.0 / h_in
    R_wall = (d_out * np.log(d_out / d_in)) / (2.0 * k_tube)
    U = 1.0 / (R_in + R_wall + R_f_out + 1.0 / h_out)
    NTU = U * A_coil / (m_dot * cp_c)
    eps = 1.0 - np.exp(-NTU)
    return eps * m_dot * cp_c * (T_water_in - T_band)


# ------------------------------------------------------------
# STATE HELPERS (enthalpy -> temperature, phase classification)
# ------------------------------------------------------------
def H_to_T(H, mat):
    """Enthalpy (scalar or array) -> temperature (degC)."""
    return np.interp(H, mat["H_grid"], mat["T_grid"])


def classify(cfg, mat, H, m_shell):
    """Split the tank into liquid / slurry / solid mass and mass-average T."""
    N = cfg["N"]
    liq = slu = sol = 0.0
    liqT = sluT = solT = 0.0
    for i in range(N):
        T = float(np.interp(H[i], mat["H_grid"], mat["T_grid"]))
        m = m_shell[i]
        if T > cfg["T_me"]:
            liq += m; liqT += m * T
        elif T >= cfg["T_ms"]:
            slu += m; sluT += m * T
        else:
            sol += m; solT += m * T
    return dict(
        liquid=liq, slurry=slu, solid=sol,
        liquid_T=liqT / liq if liq > 0 else 0.0,
        slurry_T=sluT / slu if slu > 0 else 0.0,
        solid_T=solT / sol if sol > 0 else 0.0,
    )


def avg_T(r):
    tot = r["liquid"] + r["slurry"] + r["solid"]
    return ((r["liquid"] * r["liquid_T"] + r["slurry"] * r["slurry_T"]
             + r["solid"] * r["solid_T"]) / tot if tot > 0 else 0.0)


def pack_result(cfg, mat, R_coil, n_turns, H_out, T_steel, m_shell, T_initial,
                peak_oil, Q_total, peak_Q):
    """Result record shared by heating runs and cooldown runs."""
    st = classify(cfg, mat, H_out, m_shell)
    if T_initial is not None:
        st0 = classify(cfg, mat, np.array(T_initial["H"]), m_shell)
    else:
        st0 = dict(liquid=0.0, slurry=0.0, solid=0.0,
                   liquid_T=0.0, slurry_T=0.0, solid_T=0.0)
    L_coil = 2.0 * np.pi * R_coil * n_turns
    return dict(
        R=R_coil, n=int(n_turns), L_coil=L_coil,
        A_coil=np.pi * cfg["d_out"] * L_coil,
        liquid_0=st0["liquid"], slurry_0=st0["slurry"], solid_0=st0["solid"],
        liquid=st["liquid"], slurry=st["slurry"], solid=st["solid"],
        liquid_T=st["liquid_T"], slurry_T=st["slurry_T"], solid_T=st["solid_T"],
        peak_oil=peak_oil, energy_kWh=Q_total / 3.6e6,
        peak_power_kW=peak_Q / 1000.0,
        H_final=H_out.tolist(), T_steel=float(T_steel),
    )


def wall_U_at_T(cfg, T):
    k_wall = float(np.interp(T, cfg["T_wall_arr"], cfg["k_wall_arr"]))
    R_wall_half = (cfg["t_wall"] / 2.0) / k_wall
    R_ins = (cfg["t_ins"] / cfg["k_ins"]) if cfg["t_ins"] > 0 else 0.0
    R_conv = 1.0 / cfg["h_air"]
    return 1.0 / (R_wall_half + R_ins + R_conv)
