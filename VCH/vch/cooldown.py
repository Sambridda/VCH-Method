"""Cooldown: radial conduction with no coil power."""
import numpy as np
from .physics import _jit, pack_result
from .geometry import (prepare_mesh, injection_weights, steel_mass, tank_outer_area)
from .sweep import Cancelled, _check

@_jit
def cooldown_kernel(H, T_steel, N, dr, dt_inv_m_eff, A_face,
                    T_grid, H_grid, k_grid,
                    M_steel,
                    T_wall_arr, k_wall_arr, cp_wall_arr,
                    A_tank_outer, t_wall, t_ins, k_ins, h_air,
                    T_amb, dt, n_steps):
    T = np.empty(N)
    k_arr = np.empty(N)
    for step in range(n_steps):
        for i in range(N):
            T[i] = np.interp(H[i], H_grid, T_grid)
        for i in range(N):
            k_arr[i] = np.interp(T[i], T_grid, k_grid)
        for i in range(N - 1):
            kavg = 0.5 * (k_arr[i] + k_arr[i + 1])
            Qc = kavg * A_face[i] * (T[i] - T[i + 1]) / dr
            H[i] -= Qc * dt_inv_m_eff[i]
            H[i + 1] += Qc * dt_inv_m_eff[i + 1]
        Q_os = k_arr[N - 1] * A_tank_outer * (T[N - 1] - T_steel) / (dr * 0.5)
        H[N - 1] -= Q_os * dt_inv_m_eff[N - 1]

        k_wall_now = np.interp(T_steel, T_wall_arr, k_wall_arr)
        cp_wall_now = np.interp(T_steel, T_wall_arr, cp_wall_arr)
        R_conv = 1.0 / h_air
        R_wall_half = (t_wall / 2.0) / k_wall_now
        R_ins = (t_ins / k_ins) if t_ins > 0.0 else 0.0
        U_air = 1.0 / (R_wall_half + R_ins + R_conv)
        Q_sa = U_air * A_tank_outer * (T_steel - T_amb)
        T_steel += (Q_os - Q_sa) * dt / (M_steel * cp_wall_now)
    return H, T_steel

def run_cool(cfg, mat, R_coil, n_turns, T_amb, T_initial, t_hr):
    if T_initial is None:
        raise ValueError("run_cool needs a start state (the cold-run end state); got None.")
    N = cfg["N"]
    T_ref_mesh = float(T_initial["T_steel"])
    dr, r_in, r_out, r_mid, A_face, m_shell = prepare_mesh(cfg, mat, T_ref_mesh)
    inj_w = injection_weights(R_coil, cfg["d_out"], N, dr)

    H = np.array(T_initial["H"], dtype=float)
    T_steel = float(T_initial["T_steel"])

    M_steel = steel_mass(cfg, 0.5 * (cfg["T_amb_low"] + cfg["T_water_in"]))
    A_outer = tank_outer_area(cfg)
    dt = cfg["dt_sec"]
    n_steps = int(round(t_hr * 3600.0 / dt))

    L_coil = 2.0 * np.pi * R_coil * n_turns
    V_metal = np.pi * 0.25 * (cfg["d_out"] ** 2 - cfg["d_in"] ** 2) * L_coil
    V_water = np.pi * 0.25 * cfg["d_in"] ** 2 * L_coil
    rho_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["rho_tube_arr"]))
    cp_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["cp_tube_arr"]))
    rho_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    cp_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["cp_water_arr"]))
    M_coil_metal = rho_tube_ref * V_metal
    M_coil_water = rho_water_ref * V_water
    cp_oil_ref = float(np.interp(cfg["T_water_in"], mat["T_grid"], mat["cp_sens_grid"]))
    extra_kg = (M_coil_metal * cp_tube_ref + M_coil_water * cp_water_ref) / cp_oil_ref
    m_eff = m_shell + extra_kg * inj_w
    dt_inv_m_eff = dt / m_eff

    H_out, T_steel = cooldown_kernel(
        H, T_steel, N, dr, dt_inv_m_eff, A_face,
        mat["T_grid"], mat["H_grid"], mat["k_grid"],
        M_steel,
        cfg["T_wall_arr"], cfg["k_wall_arr"], cfg["cp_wall_arr"],
        A_outer, cfg["t_wall"], cfg["t_ins"], cfg["k_ins"], cfg["h_air"],
        T_amb, dt, n_steps)

    return pack_result(cfg, mat, R_coil, n_turns, H_out, T_steel, m_shell,
                       T_initial, 0.0, 0.0, 0.0)

def run_cooldowns(cfg, mat, R, n, presets, cold_rows, progress=None, cancel=None):
    cold_at_selected = {}; cool_states = {}; cool_reports = {}
    total = len(presets)
    for k, (pname, Tamb) in enumerate(presets):
        _check(cancel)
        cold_row = next(r for r in cold_rows
                        if r["preset"] == pname
                        and abs(r["R"] - R) < 1e-9 and r["n"] == n)
        cold_at_selected[pname] = cold_row
        T_ini = {"H": cold_row["H_final"], "T_steel": cold_row["T_steel"]}
        r = run_cool(cfg, mat, R, n, Tamb, T_ini, cfg["t_cooldown_hr"])
        cool_states[pname] = {"H": r["H_final"], "T_steel": r["T_steel"]}
        cool_reports[pname] = r
        if progress: progress(k + 1, total, f"cooldown {pname}")
    return cold_at_selected, cool_states, cool_reports