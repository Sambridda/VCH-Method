# ============================================================
# engine_2d.py — 2D Draining Physics Engine
# ============================================================
import numpy as np
from numba import njit, objmode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys

@njit(cache=True)
def h_internal(m_dot, d_in, d_coil_actual, rho, cp, mu, k):
    A = np.pi * (d_in/2.0)**2
    v = m_dot / (max(1e-6, rho) * A)
    Re = rho * v * d_in / max(1e-6, mu)
    Pr = mu * cp / max(1e-6, k)
    if Re > 10000.0:
        f = (0.79*np.log(Re) - 1.64)**-2
        Nu = (f/8.0)*(Re - 1000.0)*Pr/(1.0 + 2.17*np.sqrt(f/8.0)*(Pr**(2.0/3.0) - 1.0))
        Nu *= (1.0 + (d_in / d_coil_actual)**0.9)
    else:
        Nu = 4.36
    return Nu * k / d_in

@njit(cache=True)
def h_external(T_oil, T_grid, rho_grid, cp_grid, k_grid, mu_grid, T_ms,
               has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural, d_out):
    if T_oil < T_ms and has_agitator == 0:
        return 5.0
    rho = np.interp(T_oil, T_grid, rho_grid)
    cp  = np.interp(T_oil, T_grid, cp_grid)
    k   = np.interp(T_oil, T_grid, k_grid)
    mu  = np.interp(T_oil, T_grid, mu_grid)
    
    if has_agitator == 1:
        N_rps = N_rpm / 60.0
        Re_ia = rho * N_rps * (D_imp**2) / max(1e-6, mu)
        Pr = mu * cp / max(1e-6, k)
        if ag_type == 1: C, a = 0.60, 0.67
        elif ag_type == 2: C, a = 0.54, 0.67
        elif ag_type == 3: C, a = 0.40, 0.50
        else: C, a = 0.35, 0.50
        Nu = C * (Re_ia ** a) * (Pr ** (1.0/3.0)) * visc_corr
        return Nu * k / d_out
    else:
        nu = mu / max(1e-6, rho); alpha = k / max(1e-6, rho * cp); Pr = nu / max(1e-6, alpha)
        d_char = d_out
        Ra = 9.81 * beta * dT_natural * (d_char**3) / max(1e-6, nu * alpha)
        Nu = (0.60 + (0.387 * Ra**(1.0/6.0)) / (1.0 + (0.559/max(1e-6, Pr))**(9.0/16.0))**(8.0/27.0))**2
        return Nu * k / d_char

@njit(cache=True)
def calc_Q(m_dot, T_band, T_water_in, d_in, d_out, R_f_out, A_coil, d_coil_actual,
           T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
           T_tube_arr, k_tube_arr,
           T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms,
           has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural):
    if T_band >= T_water_in: return 0.0
    T_eff = 0.5*(T_water_in + T_band)
    rho_c = np.interp(T_eff, T_water_arr, rho_water_arr)
    cp_c  = np.interp(T_eff, T_water_arr, cp_water_arr)
    mu_c  = np.interp(T_eff, T_water_arr, mu_water_arr)
    k_c   = np.interp(T_eff, T_water_arr, k_water_arr)
    k_tube = np.interp(T_eff, T_tube_arr, k_tube_arr)

    h_in  = h_internal(m_dot, d_in, d_coil_actual, rho_c, cp_c, mu_c, k_c)
    h_out = h_external(T_band, T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms,
                       has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural, d_out)
                       
    R_in = 1.0/h_in
    R_wall = (d_out*np.log(d_out/d_in))/(2.0*k_tube)
    U = 1.0/(R_in + R_wall + R_f_out + 1.0/h_out)
    NTU = U*A_coil/(m_dot*cp_c)
    eps = 1.0 - np.exp(-NTU)
    return eps * m_dot * cp_c * (T_water_in - T_band)

@njit(cache=True)
def kernel_2d(mass, enthalpy, N_r, N_z, dr, dz, 
              coil_w, R_coil, A_coil_total, d_in, d_out, R_f_out, m_dot_coil,
              drain_r, drain_z, drain_area, has_makeup, makeup_T,
              T_water_in, T_ms, T_me, Hfus,
              T_grid, H_grid, rho_grid, cp_sens_grid, k_grid, mu_grid,
              T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
              T_tube_arr, k_tube_arr, 
              dt, n_steps, n_steps_heat, n_steps_cool, n_cycles, T_amb, U_tank_air,
              K_settle,
              has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural):
    
    drained_mass_cum = 0.0
    filled_mass_cum = 0.0
    idle_time = 0.0
    drain_log = np.zeros(n_steps)
    fill_log = np.zeros(n_steps)
    time_log = np.zeros(n_steps)
    is_heating_log = np.zeros(n_steps, dtype=np.bool_)
    
    cycle_extracted = np.zeros(n_cycles)
    cycle_idle_hr = np.zeros(n_cycles)
    cycle_produced = np.zeros(n_cycles)
    cycle_stranded = np.zeros(n_cycles)
    
    cyc_sol_kg = np.zeros(n_cycles); cyc_sol_T = np.zeros(n_cycles)
    cyc_slu_kg = np.zeros(n_cycles); cyc_slu_T = np.zeros(n_cycles)
    cyc_liq_kg = np.zeros(n_cycles); cyc_liq_T = np.zeros(n_cycles)
    
    T_snapshots = np.full((n_cycles, N_r, N_z), np.nan)
    M_snapshots = np.full((n_cycles, N_r, N_z), np.nan)
    
    max_hours = 48
    hourly_T_snaps = np.full((n_cycles, max_hours, N_r, N_z), np.nan)
    hourly_M_snaps = np.full((n_cycles, max_hours, N_r, N_z), np.nan)
    
    V_cell = np.empty((N_r, N_z))
    for i in range(N_r):
        r_in = i * dr; r_out = (i + 1) * dr
        for j in range(N_z):
            V_cell[i, j] = np.pi * (r_out**2 - r_in**2) * dz
            
    cycle_len = max(1, n_steps_heat + n_steps_cool)
    T_soft = T_ms - 10.0 

    stranded_prev = 0.0
    for i in range(N_r):
        for j in range(N_z):
            if mass[i, j] > 1e-6:
                T_curr = np.interp(enthalpy[i, j] / mass[i, j], H_grid, T_grid)
                if T_curr > T_me:
                    stranded_prev += mass[i, j]

    extracted_so_far = 0.0
    filled_so_far = 0.0
    idle_so_far = 0.0

    for step in range(n_steps):
        if step % 100 == 0 or step == n_steps - 1: 
            with objmode():
                pct = (step / max(1, n_steps)) * 100.0
                sys.stdout.write("\r  2D Simulating... %.1f%% " % pct)
                sys.stdout.flush()

        time_log[step] = step * dt / 3600.0
        
        step_in_cycle = step % cycle_len
        cycle_idx = step // cycle_len
        is_heating = (step_in_cycle < n_steps_heat)
        is_heating_log[step] = is_heating
        
        T = np.zeros((N_r, N_z))
        for i in range(N_r):
            for j in range(N_z):
                if mass[i, j] > 1e-6:
                    T[i, j] = np.interp(enthalpy[i, j] / mass[i, j], H_grid, T_grid)
                else:
                    T[i, j] = T_amb 

        steps_per_hour = int(3600 / dt)
        if is_heating and (step_in_cycle % steps_per_hour == 0):
            hour_idx = step_in_cycle // steps_per_hour
            if hour_idx < max_hours:
                hourly_T_snaps[cycle_idx, hour_idx, :, :] = T
                hourly_M_snaps[cycle_idx, hour_idx, :, :] = mass

        surf = np.zeros(N_r, dtype=np.int64)
        for i in range(N_r):
            surf[i] = 0
            for j in range(N_z - 1, -1, -1):
                if mass[i, j] > 1e-6:
                    surf[i] = j
                    break
                    
        avg_surf = np.mean(surf)

        if is_heating:
            T_coil_avg = 0.0; weight_sum = 0.0
            for i in range(N_r):
                for j in range(N_z):
                    if coil_w[i, j] > 1e-6 and mass[i, j] > 1e-6:
                        T_coil_avg += T[i, j] * coil_w[i, j]
                        weight_sum += coil_w[i, j]
                        
            if weight_sum > 0.0:
                T_coil_avg /= weight_sum
                d_coil_actual = 2.0 * R_coil
                q_total = calc_Q(m_dot_coil, T_coil_avg, T_water_in, d_in, d_out, R_f_out, A_coil_total, d_coil_actual,
                                 T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
                                 T_tube_arr, k_tube_arr, T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms,
                                 has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural)
                
                for i in range(N_r):
                    for j in range(N_z):
                        if coil_w[i, j] > 1e-6 and mass[i, j] > 1e-6:
                            enthalpy[i, j] += q_total * (coil_w[i, j] / weight_sum) * dt

        for i in range(N_r):
            for j in range(N_z):
                if mass[i, j] <= 1e-6: continue
                k_here = np.interp(T[i, j], T_grid, k_grid)
                
                if i > 0 and mass[i-1, j] > 1e-6:
                    k_avg = 0.5 * (k_here + np.interp(T[i-1, j], T_grid, k_grid))
                    area = 2.0 * np.pi * (i * dr) * dz
                    q = k_avg * area * (T[i, j] - T[i-1, j]) / dr
                    enthalpy[i, j] -= q * dt
                if i < N_r - 1 and mass[i+1, j] > 1e-6:
                    k_avg = 0.5 * (k_here + np.interp(T[i+1, j], T_grid, k_grid))
                    area = 2.0 * np.pi * ((i+1) * dr) * dz
                    q = k_avg * area * (T[i, j] - T[i+1, j]) / dr
                    enthalpy[i, j] -= q * dt
                    
                if j > 0 and mass[i, j-1] > 1e-6:
                    k_avg = 0.5 * (k_here + np.interp(T[i, j-1], T_grid, k_grid))
                    if T[i, j-1] > T[i, j]: k_avg *= 50.0
                    area = 2.0 * np.pi * (i * dr) * dr
                    q = k_avg * area * (T[i, j] - T[i, j-1]) / dz
                    enthalpy[i, j] -= q * dt
                if j < N_z - 1 and mass[i, j+1] > 1e-6:
                    k_avg = 0.5 * (k_here + np.interp(T[i, j+1], T_grid, k_grid))
                    if T[i, j] > T[i, j+1]: k_avg *= 50.0
                    area = 2.0 * np.pi * (i * dr) * dr
                    q = k_avg * area * (T[i, j] - T[i, j+1]) / dz
                    enthalpy[i, j] -= q * dt

        drained_this_step = 0.0
        if is_heating:
            surf_drain = -1
            for j in range(N_z - 1, drain_z - 1, -1):
                if mass[drain_r, j] > 1e-6 and T[drain_r, j] > T_me:
                    surf_drain = j
                    break
            
            if surf_drain > drain_z:
                h_head = (surf_drain - drain_z) * dz
                rho_liq = np.interp(T_me, T_grid, rho_grid)
                v_flow = 0.6 * np.sqrt(2.0 * 9.81 * h_head)
                target_mass = rho_liq * v_flow * drain_area * dt
                
                R_suction_float = avg_surf * (dz / dr)
                R_suction = max(1, int(R_suction_float))
                
                i_start = drain_r
                i_end = max(0, drain_r - R_suction)
                
                w_array = np.zeros(N_r)
                j_array = np.zeros(N_r, dtype=np.int64)
                total_weight = 0.0
                
                for i in range(i_start, i_end - 1, -1):
                    best_j = -1
                    for j in range(drain_z, N_z):
                        if mass[i, j] > 1e-6 and T[i, j] > T_soft:
                            best_j = j
                            break
                    
                    if best_j != -1:
                        dist = abs(i - drain_r) * dr
                        max_dist = R_suction * dr
                        w = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
                        if w > 0.0:
                            w_array[i] = w
                            j_array[i] = best_j
                            total_weight += w
                            
                if total_weight > 0.0:
                    for i in range(i_start, i_end - 1, -1):
                        if w_array[i] > 0.0:
                            dm = target_mass * (w_array[i] / total_weight)
                            actual_dm = min(dm, mass[i, j_array[i]])
                            
                            if actual_dm > 0.0:
                                fraction = actual_dm / max(1e-6, mass[i, j_array[i]])
                                enthalpy[i, j_array[i]] *= (1.0 - fraction)
                                mass[i, j_array[i]] -= actual_dm
                                drained_mass_cum += actual_dm
                                drained_this_step += actual_dm
        
        if is_heating and drained_this_step == 0.0:
            idle_time += dt / 3600.0

        filled_this_step = 0.0
        if has_makeup == 1 and drained_this_step > 0.0:
            h_makeup = np.interp(makeup_T, T_grid, H_grid)
            j_top = N_z - 1
            for i in range(N_r):
                if mass[i, j_top] < V_cell[i, j_top]:
                    needed = V_cell[i, j_top] - mass[i, j_top]
                    added = min(needed, drained_this_step)
                    mass[i, j_top] += added
                    enthalpy[i, j_top] += added * h_makeup
                    drained_this_step -= added
                    filled_mass_cum += added
                    filled_this_step += added
                    if drained_this_step <= 1e-6: break
            
        drain_log[step] = drained_mass_cum
        fill_log[step] = filled_mass_cum
        
        if has_makeup == 0:
            for j in range(0, N_z - 1):
                for i in range(N_r):
                    if mass[i, j] < V_cell[i, j] and mass[i, j+1] > 1e-6:
                        needed = V_cell[i, j] - mass[i, j]
                        available = mass[i, j+1]
                        
                        h_head = (surf[i] - j) * dz
                        mu_above = np.interp(T[i, j+1], T_grid, mu_grid)
                        
                        mobility = (K_settle * h_head) / max(1e-6, mu_above)
                        mobility = min(1.0, max(0.01, mobility))
                        
                        moved = min(needed * mobility, available)
                        
                        if moved > 1e-9:
                            h_spec = enthalpy[i, j+1] / max(1e-6, mass[i, j+1])
                            mass[i, j] += moved; mass[i, j+1] -= moved
                            enthalpy[i, j] += moved * h_spec
                            enthalpy[i, j+1] -= moved * h_spec

        if has_makeup == 0:
            surf = np.zeros(N_r, dtype=np.int64)
            for i in range(N_r):
                surf[i] = 0
                for j in range(N_z - 1, -1, -1):
                    if mass[i, j] > 1e-6:
                        surf[i] = j
                        break

            for i in range(N_r - 1):
                if surf[i] > surf[i+1]:
                    j_top = surf[i]; j_target = surf[i+1]
                    if T[i, j_top] > T_soft:
                        moved = mass[i, j_top] * 0.5
                        h_spec = enthalpy[i, j_top] / max(1e-6, mass[i, j_top])
                        mass[i, j_top] -= moved; mass[i+1, j_target] += moved
                        enthalpy[i, j_top] -= moved * h_spec
                        enthalpy[i+1, j_target] += moved * h_spec
                elif surf[i+1] > surf[i]:
                    j_top = surf[i+1]; j_target = surf[i]
                    if T[i+1, j_top] > T_soft:
                        moved = mass[i+1, j_top] * 0.5
                        h_spec = enthalpy[i+1, j_top] / max(1e-6, mass[i+1, j_top])
                        mass[i+1, j_top] -= moved; mass[i, j_target] += moved
                        enthalpy[i+1, j_top] -= moved * h_spec
                        enthalpy[i, j_target] += moved * h_spec

            for i in range(N_r - 2, -1, -1):
                if surf[i] > surf[i+1]:
                    j_top = surf[i]; j_target = surf[i+1]
                    if T[i, j_top] > T_soft:
                        moved = mass[i, j_top] * 0.5
                        h_spec = enthalpy[i, j_top] / max(1e-6, mass[i, j_top])
                        mass[i, j_top] -= moved; mass[i+1, j_target] += moved
                        enthalpy[i, j_top] -= moved * h_spec
                        enthalpy[i+1, j_target] += moved * h_spec
                elif surf[i+1] > surf[i]:
                    j_top = surf[i+1]; j_target = surf[i]
                    if T[i+1, j_top] > T_soft:
                        moved = mass[i+1, j_top] * 0.5
                        h_spec = enthalpy[i+1, j_top] / max(1e-6, mass[i+1, j_top])
                        mass[i+1, j_top] -= moved; mass[i, j_target] += moved
                        enthalpy[i+1, j_top] -= moved * h_spec
                        enthalpy[i, j_target] += moved * h_spec

        for j in range(N_z):
            if mass[N_r-1, j] > 1e-6:
                area = 2.0 * np.pi * ((N_r - 1) * dr) * dz
                q_loss = U_tank_air * area * (T[N_r-1, j] - T_amb)
                enthalpy[N_r-1, j] -= q_loss * dt

        if is_heating and (step_in_cycle == n_steps_heat - 1):
            for i in range(N_r):
                for j in range(N_z):
                    if mass[i, j] <= 1e-6:
                        T_snapshots[cycle_idx, i, j] = np.nan
                        M_snapshots[cycle_idx, i, j] = np.nan
                    else:
                        T_snapshots[cycle_idx, i, j] = T[i, j]
                        M_snapshots[cycle_idx, i, j] = mass[i, j]
            
            liquid_now = 0.0
            for i in range(N_r):
                for j in range(N_z):
                    if T[i, j] > T_me:
                        liquid_now += mass[i, j]
            
            cycle_stranded[cycle_idx] = liquid_now
            cycle_extracted[cycle_idx] = drained_mass_cum - extracted_so_far
            cycle_idle_hr[cycle_idx] = idle_time - idle_so_far
            cycle_produced[cycle_idx] = cycle_extracted[cycle_idx] + (liquid_now - stranded_prev)
            
            extracted_so_far = drained_mass_cum
            idle_so_far = idle_time
            stranded_prev = liquid_now

            m_sol = 0.0; T_sol_sum = 0.0
            m_slu = 0.0; T_slu_sum = 0.0
            m_liq = 0.0; T_liq_sum = 0.0
            for i in range(N_r):
                for j in range(N_z):
                    if mass[i, j] > 1e-6:
                        if T[i, j] < T_ms:
                            m_sol += mass[i, j]; T_sol_sum += mass[i, j] * T[i, j]
                        elif T[i, j] <= T_me:
                            m_slu += mass[i, j]; T_slu_sum += mass[i, j] * T[i, j]
                        else:
                            m_liq += mass[i, j]; T_liq_sum += mass[i, j] * T[i, j]
            
            cyc_sol_kg[cycle_idx] = m_sol
            cyc_slu_kg[cycle_idx] = m_slu
            cyc_liq_kg[cycle_idx] = m_liq
            cyc_sol_T[cycle_idx] = T_sol_sum / max(1e-6, m_sol)
            cyc_slu_T[cycle_idx] = T_slu_sum / max(1e-6, m_slu)
            cyc_liq_T[cycle_idx] = T_liq_sum / max(1e-6, m_liq)

    with objmode():
        sys.stdout.write("\r  2D Simulating... 100.0%  \n")
        sys.stdout.flush()

    return (time_log, drain_log, fill_log, drained_mass_cum, filled_mass_cum, idle_time, T, mass, is_heating_log, 
            T_snapshots, M_snapshots, cycle_extracted, cycle_idle_hr, cycle_produced, cycle_stranded,
            cyc_sol_kg, cyc_slu_kg, cyc_liq_kg, cyc_sol_T, cyc_slu_T, cyc_liq_T,
            hourly_T_snaps, hourly_M_snaps)

# ------------------------------------------------------------
# DRIVER FUNCTION
# ------------------------------------------------------------
def run_2d_simulation(cfg, mat, R_coil, n_turns, initial_1d_state, drain_diameter, drain_location, outdir):
    print(f"\n[Engine 2D] Initializing 2D Grid for Draining Simulation...")
        
    res = 5 
    N_r = 10 * res
    N_z = 10 * res
    
    tank_height = max(1e-6, cfg["tank_height"]) 
    dr = max(1e-6, cfg["r_inner"] / N_r)
    dz = max(1e-6, tank_height / N_z)
    
    mass = np.zeros((N_r, N_z))
    enthalpy = np.zeros((N_r, N_z))
    
    rho_amb = max(1e-6, np.interp(cfg["T_amb_mid"], mat["T_grid"], mat["rho_grid"]))
    h_spec_amb = np.interp(cfg["T_amb_mid"], mat["T_grid"], mat["H_grid"])
    
    V_total_tank = np.pi * cfg["r_inner"]**2 * tank_height
    V_target = cfg["M_fluid"] / rho_amb
    fill_fraction = V_target / max(1e-6, V_total_tank)
    
    if fill_fraction > 1.0: fill_fraction = 1.0
    
    j_fill_float = fill_fraction * N_z
    j_fill_int = int(j_fill_float)
    j_fill_partial = j_fill_float - j_fill_int

    for i in range(N_r):
        r_in = i * dr; r_out = (i + 1) * dr
        V_cell = np.pi * (r_out**2 - r_in**2) * dz
        
        for j in range(N_z):
            if j < j_fill_int:
                mass[i, j] = V_cell * rho_amb
                enthalpy[i, j] = mass[i, j] * h_spec_amb
            elif j == j_fill_int:
                mass[i, j] = V_cell * rho_amb * j_fill_partial
                enthalpy[i, j] = mass[i, j] * h_spec_amb
            else:
                mass[i, j] = 0.0
                enthalpy[i, j] = 0.0

    coil_w = np.zeros((N_r, N_z))
    r_lo = R_coil - cfg["d_out"]/2.0
    r_hi = R_coil + cfg["d_out"]/2.0
    z_lo = cfg["coil_clearance_bottom"]
    z_hi = cfg["coil_clearance_bottom"] + cfg["L_avail"]
    
    total_w = 0.0
    for i in range(N_r):
        c_r_lo = i * dr; c_r_hi = (i+1) * dr
        ov_r = max(0.0, min(c_r_hi, r_hi) - max(c_r_lo, r_lo))
        if ov_r <= 0: continue
        
        for j in range(N_z):
            c_z_lo = j * dz; c_z_hi = (j+1) * dz
            ov_z = max(0.0, min(c_z_hi, z_hi) - max(c_z_lo, z_lo))
            if ov_z <= 0: continue
            
            w = ov_r * ov_z
            coil_w[i, j] = w
            total_w += w
            
    if total_w > 0.0:
        for i in range(N_r):
            for j in range(N_z):
                coil_w[i, j] /= total_w

    drain_r_idx, drain_z_idx = drain_location
    
    T_drain_init = np.interp(enthalpy[drain_r_idx, drain_z_idx] / max(1e-6, mass[drain_r_idx, drain_z_idx]), mat["H_grid"], mat["T_grid"]) if mass[drain_r_idx, drain_z_idx] > 0 else cfg["T_amb_mid"]
    if T_drain_init < cfg["T_me"]:
        print(f"  WARNING: The drain cell is currently SOLID ({T_drain_init:.1f} °C). It will idle until heat reaches it.")
    
    drain_area = np.pi * (drain_diameter / 2.0)**2
    A_coil_total = 2.0 * np.pi * R_coil * n_turns * cfg["d_out"]
    
    k_wall = float(np.interp(cfg["T_water_in"], cfg["T_wall_arr"], cfg["k_wall_arr"]))
    R_wall_half = (cfg["t_wall"]/2.0) / max(1e-6, k_wall)
    R_ins = (cfg["t_ins"]/max(1e-6, cfg["k_ins"])) if cfg["t_ins"] > 0 else 0.0
    R_conv = 1.0 / max(1e-6, cfg["h_air"])
    U_tank_air = 1.0 / max(1e-6, R_wall_half + R_ins + R_conv)
    
    n_cycles = max(1, cfg["N_cycles"])
    n_steps_heat = max(1, int(cfg["t_window_hr"] * 3600.0 / max(1e-6, cfg["dt_sec"])))
    n_steps_cool = max(1, int(cfg["t_cooldown_hr"] * 3600.0 / max(1e-6, cfg["dt_sec"])))
    total_steps = n_cycles * n_steps_heat + (n_cycles - 1) * n_steps_cool
    total_hr = total_steps * cfg["dt_sec"] / 3600.0
    
    (time_log, drain_log, fill_log, total_drained, total_filled, idle_hr, final_T, final_mass, 
     is_heating_log, T_snapshots, M_snapshots, cycle_extracted, cycle_idle, cycle_produced, cycle_stranded,
     cyc_sol_kg, cyc_slu_kg, cyc_liq_kg, cyc_sol_T, cyc_slu_T, cyc_liq_T,
     hourly_T_snaps, hourly_M_snaps) = kernel_2d(
        mass, enthalpy, N_r, N_z, dr, dz, 
        coil_w, R_coil, A_coil_total, cfg["d_in"], cfg["d_out"], cfg["R_f_out"], 
        cfg["flow_m3h"] * float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"])) / 3600.0,
        drain_r_idx, drain_z_idx, drain_area, cfg["has_makeup"], cfg["makeup_T"],
        cfg["T_water_in"], cfg["T_ms"], cfg["T_me"], cfg["Hfus"],
        mat["T_grid"], mat["H_grid"], mat["rho_grid"], mat["cp_sens_grid"], mat["k_grid"], mat["mu_grid"],
        cfg["T_water_arr"], cfg["rho_water_arr"], cfg["cp_water_arr"], cfg["mu_water_arr"], cfg["k_water_arr"],
        cfg["T_tube_arr"], cfg["k_tube_arr"],
        cfg["dt_sec"], total_steps, n_steps_heat, n_steps_cool, n_cycles, cfg["T_amb_mid"], U_tank_air,
        cfg["K_settle"],
        cfg["has_agitator"], cfg["ag_type"], cfg["N_rpm"], cfg["D_imp"], cfg["visc_corr"], cfg["beta_liq"], cfg["dT_natural"]
    )

    # 1. Extraction Curve
    curve_path = os.path.join(outdir, f"b1_drain_curve.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_log, drain_log, linewidth=2, color="tab:blue", label="Extracted Liquid")
    
    for step in range(len(time_log)):
        if not is_heating_log[step]:
            ax.axvspan(time_log[step], time_log[step] + cfg["dt_sec"]/3600.0, color='grey', alpha=0.2, lw=0)
    
    legend_elements = [
        plt.Line2D([0], [0], color='tab:blue', lw=2, label='Extracted Liquid'),
        Patch(facecolor='grey', alpha=0.2, label='Cooling Phase (Coil OFF)')
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative Liquid Extracted (kg)")
    ax.set_title(f"Extraction Curve ({n_cycles} Cycles, Valve D={drain_diameter*1000:.1f} mm)")
    ax.grid(True, alpha=0.3)
    fig.savefig(curve_path, dpi=130); plt.close(fig)

    # 2. Heatmaps, Mass Maps, Phase Bars, and Hourly Snapshots
    heatmap_paths = []
    massmap_paths = []
    phase_bar_paths = []
    
    all_hourly_heatmaps = []
    all_hourly_massmaps = []
    
    for c in range(n_cycles):
        r_grid = np.linspace(0, cfg["r_inner"], N_r)
        z_grid = np.linspace(0, tank_height, N_z)
        
        r_wall_start = cfg["r_inner"]
        r_wall_end = cfg["r_inner"] + cfg["t_wall"]
        r_ins_end = r_wall_end + cfg["t_ins"]
        
        # Heatmap
        h_path = os.path.join(outdir, f"b1_heatmap_cyc{c+1}.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.inferno.copy(); cmap.set_bad('white') 
        c_mesh = ax.pcolormesh(r_grid, z_grid, T_snapshots[c].T, cmap=cmap, shading='auto')
        
        ax.axvspan(r_wall_start, r_wall_end, color='dimgrey', alpha=0.8, lw=0)
        if cfg["t_ins"] > 0:
            ax.axvspan(r_wall_end, r_ins_end, color='lightgrey', alpha=0.8, lw=0, hatch='//')
            
        fig.colorbar(c_mesh, ax=ax, label='Temperature (°C)')
        ax.set_xlabel("Radius (m)"); ax.set_ylabel("Height (m)")
        ax.set_title(f"Cycle {c+1} — Temp (with Wall & Insulation)")
        ax.set_xlim(0, r_ins_end)
        ax.axhline(cfg["coil_clearance_bottom"], color='cyan', linestyle='--', linewidth=1)
        ax.axhline(cfg["coil_clearance_bottom"] + cfg["L_avail"], color='cyan', linestyle='--', linewidth=1)
        fig.savefig(h_path, dpi=130); plt.close(fig)
        heatmap_paths.append(h_path)

        # Mass Map
        m_path = os.path.join(outdir, f"b1_massmap_cyc{c+1}.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap_m = plt.cm.Blues.copy(); cmap_m.set_bad('white')
        m_mesh = ax.pcolormesh(r_grid, z_grid, M_snapshots[c].T, cmap=cmap_m, shading='auto')
        
        ax.axvspan(r_wall_start, r_wall_end, color='dimgrey', alpha=0.8, lw=0)
        if cfg["t_ins"] > 0:
            ax.axvspan(r_wall_end, r_ins_end, color='lightgrey', alpha=0.8, lw=0, hatch='//')
            
        fig.colorbar(m_mesh, ax=ax, label='Mass (kg)')
        ax.set_xlabel("Radius (m)"); ax.set_ylabel("Height (m)")
        ax.set_title(f"Cycle {c+1} — Mass (with Wall & Insulation)")
        ax.set_xlim(0, r_ins_end)
        ax.axhline(cfg["coil_clearance_bottom"], color='red', linestyle='--', linewidth=1)
        ax.axhline(cfg["coil_clearance_bottom"] + cfg["L_avail"], color='red', linestyle='--', linewidth=1)
        fig.savefig(m_path, dpi=130); plt.close(fig)
        massmap_paths.append(m_path)

        # Phase Bar Chart
        pb_path = os.path.join(outdir, f"b1_phasebar_cyc{c+1}.png")
        fig, ax = plt.subplots(figsize=(6, 4))
        phases = ['Solid', 'Slurry', 'Liquid']
        masses = [cyc_sol_kg[c], cyc_slu_kg[c], cyc_liq_kg[c]]
        temps = [cyc_sol_T[c], cyc_slu_T[c], cyc_liq_T[c]]
        colors = ['tab:gray', 'tab:orange', 'tab:blue']
        
        bars = ax.bar(phases, masses, color=colors)
        for idx, bar in enumerate(bars):
            if masses[idx] > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(masses)*0.02, 
                        f"{masses[idx]:.0f} kg\n{temps[idx]:.1f} °C", ha='center', fontsize=9)
                
        ax.set_ylabel("Mass (kg)"); ax.set_title(f"Cycle {c+1} — Phase Distribution")
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(pb_path, dpi=130); plt.close(fig)
        phase_bar_paths.append(pb_path)

        # --- Hourly Snapshots ---
        cyc_hourly_heatmaps = []
        cyc_hourly_massmaps = []
        
        for h in range(48): 
            if np.all(np.isnan(hourly_T_snaps[c, h])):
                break 
                
            hh_path = os.path.join(outdir, f"b1_hourly_heatmap_cyc{c+1}_h{h+1}.png")
            hm_path = os.path.join(outdir, f"b1_hourly_massmap_cyc{c+1}_h{h+1}.png")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            c_mesh = ax.pcolormesh(r_grid, z_grid, hourly_T_snaps[c, h].T, cmap=cmap, shading='auto')
            fig.colorbar(c_mesh, ax=ax, label='Temperature (°C)')
            ax.set_title(f"Cycle {c+1} - Hour {h+1} Temp")
            ax.set_xlabel("Radius (m)"); ax.set_ylabel("Height (m)")
            fig.savefig(hh_path, dpi=100); plt.close(fig)
            cyc_hourly_heatmaps.append(hh_path)

            fig, ax = plt.subplots(figsize=(10, 8))
            m_mesh = ax.pcolormesh(r_grid, z_grid, hourly_M_snaps[c, h].T, cmap=cmap_m, shading='auto')
            fig.colorbar(m_mesh, ax=ax, label='Mass (kg)')
            ax.set_title(f"Cycle {c+1} - Hour {h+1} Mass")
            ax.set_xlabel("Radius (m)"); ax.set_ylabel("Height (m)")
            fig.savefig(hm_path, dpi=100); plt.close(fig)
            cyc_hourly_massmaps.append(hm_path)

        all_hourly_heatmaps.append(cyc_hourly_heatmaps)
        all_hourly_massmaps.append(cyc_hourly_massmaps)

    # --- Build the updated cycle metrics for the UI and HTML ---
    cycle_metrics = []
    start_mass_cycle = cfg.get("M_fluid_effective", cfg["M_fluid"])
    
    for c in range(n_cycles):
        eff = (cycle_extracted[c] / cycle_produced[c] * 100.0) if cycle_produced[c] > 0 else 0.0
        idle_p = (cycle_idle[c] / max(1e-6, cfg["t_window_hr"]) * 100.0) if cfg["t_window_hr"] > 0 else 0.0
        
        ext_mass = cycle_extracted[c]
        end_mass = start_mass_cycle - ext_mass
        drain_util = (ext_mass / max(1e-6, start_mass_cycle)) * 100.0
        
        cycle_metrics.append({
            "cycle": c + 1,
            "start_mass": start_mass_cycle,
            "end_mass": end_mass,
            "extracted_kg": ext_mass,
            "extracted_temp": float(cyc_liq_T[c]),
            "drain_efficiency": eff,
            "drain_util": drain_util,
            "fill_mass": 0.0, 
            "fill_util": 0.0, 
            "produced_kg": cycle_produced[c],
            "stranded_kg": cycle_stranded[c],
            "idle_pct": idle_p
        })
        
        start_mass_cycle = end_mass

    # --- GENERATE FINAL SUMMARY PLOTS ---
    
    # 1. Mass Flow Curve
    flow_path = os.path.join(outdir, "final_mass_flow.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_log, drain_log, linewidth=2, color="tab:blue", label="Extracted (kg)")
    if np.any(fill_log > 0):
        ax.plot(time_log, fill_log, linewidth=2, color="tab:orange", label="Filled (kg)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Mass (kg)")
    ax.set_title(f"Mass Flow Curve ({n_cycles} Cycles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(flow_path, dpi=130); plt.close(fig)

    # 2. Timeline (Gantt Chart) - FIXED OVERLAP
    timeline_path = os.path.join(outdir, "final_timeline.png")
    fig, ax = plt.subplots(figsize=(12, 2.5)) # Increased height
    for step in range(len(time_log) - 1):
        color = 'orange' if is_heating_log[step] else 'grey'
        ax.axvspan(time_log[step], time_log[step+1], color=color, alpha=0.8, lw=0)
        
    legend_elements = [Patch(facecolor='orange', label='Heating'), Patch(facecolor='grey', label='Cooling')]
    # Moved legend lower and added pad
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2)
    ax.set_yticks([])
    ax.set_xlabel("Time (hours)")
    fig.tight_layout(); fig.savefig(timeline_path, dpi=130); plt.close(fig)

    # 3. Drain Summary Data
    first_drain = next((time_log[i] for i in range(len(drain_log)) if drain_log[i] > 0), -1)
    last_drain = next((time_log[i] for i in reversed(range(len(drain_log))) if drain_log[i] >= total_drained - 0.1), -1)
    
    drain_summary = {
        "first_drain": float(first_drain) if first_drain >= 0 else -1,
        "last_drain": float(last_drain) if last_drain >= 0 else -1,
        "total_drained": float(total_drained),
        "total_filled": float(total_filled),
        "total_remaining": float(np.sum(mass))
    }

    return {
        "curve_path": curve_path,
        "heatmap_paths": heatmap_paths,
        "massmap_paths": massmap_paths,
        "phasebar_paths": phase_bar_paths,
        "hourly_heatmap_paths": all_hourly_heatmaps,
        "hourly_massmap_paths": all_hourly_massmaps,
        "total_hr": total_hr,
        "total_extracted_kg": total_drained,
        "cycle_metrics": cycle_metrics,
        "final_mass_flow_path": flow_path,
        "final_timeline_path": timeline_path,
        "drain_summary": drain_summary,
        
        "first_drain": float(first_drain) if first_drain >= 0 else -1,
        "last_drain": float(last_drain) if last_drain >= 0 else -1,
        "final_mass": float(np.sum(mass))
    }

def warm_up_2d():
    """Pre-compiles the Numba JIT functions so the first run isn't slow."""
    print("Warming up 2D Numba kernel...")
    kernel_2d(
        np.zeros((2,2)), np.zeros((2,2)), 2, 2, 0.1, 0.1, 
        np.zeros((2,2)), 0.5, 1.0, 0.04, 0.05, 0.0004, 0.1,
        0, 0, 0.01, 0, 20.0, 70.0, 30.0, 40.0, 65000.0,
        np.array([20.0, 30.0, 40.0]), np.array([0.0, 100.0, 200.0]), 
        np.array([900.0, 900.0, 900.0]), np.array([2000.0, 2000.0, 2000.0]), 
        np.array([0.2, 0.2, 0.2]), np.array([0.1, 0.1, 0.1]),
        np.array([20.0, 30.0, 40.0]), np.array([1000.0, 1000.0, 1000.0]), 
        np.array([4000.0, 4000.0, 4000.0]), np.array([1.0, 1.0, 1.0]), 
        np.array([0.6, 0.6, 0.6]), np.array([20.0, 30.0, 40.0]), 
        np.array([15.0, 15.0, 15.0]), 
        1.0, 10, 5, 5, 1, 20.0, 5.0, 1.0, 0, 1, 60.0, 0.5, 1.0, 0.0007, 5.0
    )