# ============================================================
# engine_2d.py — 2D Draining Physics Engine (UI Data Provider)
# ============================================================
import numpy as np
from numba import njit, objmode
import os
import sys
import outputs

# ------------------------------------------------------------
# 1D NTU HEAT TRANSFER FUNCTIONS
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# CYCLE KERNEL (Physics for ONE cycle)
# ------------------------------------------------------------
@njit(cache=True)
def cycle_kernel(
        mass, enthalpy, N_r, N_z, dr, dz, 
        coil_w, R_coil, A_coil_total, d_in, d_out, R_f_out, m_dot_coil,
        drain_r, drain_z, drain_area,
        fill_w, fill_theta_frac, fill_z, fill_mdot, fill_T, fill_mode,
        T_water_in, T_ms, T_me, Hfus,
        T_grid, H_grid, rho_grid, cp_sens_grid, k_grid, mu_grid,
        T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
        T_tube_arr, k_tube_arr, 
        dt, n_steps_heat, n_steps_cool, do_cool, n_cycles, T_amb, U_tank_air,
        K_settle, has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural,
        drained_mass_cum_in, filled_mass_cum_in, idle_time_in,
        first_drain_time_in, last_drain_time_in, stranded_prev_in,
        snapshot_interval_steps):
    
    n_steps = n_steps_heat + (n_steps_cool if do_cool else 0)
    
    time_log = np.zeros(n_steps)
    drain_log = np.zeros(n_steps)
    fill_log = np.zeros(n_steps)
    status_log = np.zeros(n_steps, dtype=np.int8)
    is_heating_log = np.zeros(n_steps, dtype=np.bool_)
    
    drained_mass_cum = drained_mass_cum_in
    filled_mass_cum = filled_mass_cum_in
    idle_time = idle_time_in
    first_drain_time = first_drain_time_in
    last_drain_time = last_drain_time_in
    stranded_prev = stranded_prev_in
    
    max_snaps = (n_steps // snapshot_interval_steps) + 2
    T_snapshots = np.full((max_snaps, N_r, N_z), np.nan)
    M_snapshots = np.full((max_snaps, N_r, N_z), np.nan)
    snapshot_times = np.zeros(max_snaps)
    snap_count = 0
    
    V_cell = np.empty((N_r, N_z))
    for i in range(N_r):
        r_in = i * dr; r_out = (i + 1) * dr
        for j in range(N_z):
            V_cell[i, j] = np.pi * (r_out**2 - r_in**2) * dz
            
    T_soft = T_ms - 10.0 
    
    cycle_start_mass = np.sum(mass)
    cycle_extracted_temp_sum = 0.0
    m_sol = 0.0; T_sol_sum = 0.0
    m_slu = 0.0; T_slu_sum = 0.0
    m_liq = 0.0; T_liq_sum = 0.0
    liquid_now = 0.0
    cycle_extracted = 0.0
    cycle_filled = 0.0
    cycle_idle_hr = 0.0
    cycle_remaining_mass = 0.0

    for step in range(n_steps):
        time_log[step] = step * dt / 3600.0
        is_heating = (step < n_steps_heat)
        is_heating_log[step] = is_heating

        T = np.zeros((N_r, N_z))
        for i in range(N_r):
            for j in range(N_z):
                if mass[i, j] > 1e-6:
                    T[i, j] = np.interp(enthalpy[i, j] / mass[i, j], H_grid, T_grid)
                else:
                    T[i, j] = T_amb 

        surf = np.zeros(N_r, dtype=np.int64)
        for i in range(N_r):
            surf[i] = 0
            for j in range(N_z - 1, -1, -1):
                if mass[i, j] > 1e-6:
                    surf[i] = j
                    break

        # 2. Coil Heating
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

        # 3. 2D Conduction
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

        # 4. Draining
        drained_this_step = 0.0
        if is_heating and drain_area > 0.0:
            if mass[drain_r, drain_z] > 1e-6 and T[drain_r, drain_z] > T_ms:
                surf_drain = drain_z
                for j in range(N_z - 1, drain_z - 1, -1):
                    if mass[drain_r, j] > 1e-6 and T[drain_r, j] > T_ms:
                        surf_drain = j
                        break
                h_head = max((surf_drain - drain_z) * dz, 0.01)
                rho_liq = np.interp(T[drain_r, drain_z], T_grid, rho_grid)
                v_flow = 0.6 * np.sqrt(2.0 * 9.81 * h_head)
                target_mass = rho_liq * v_flow * drain_area * dt
                actual_dm = min(target_mass, mass[drain_r, drain_z])
                if actual_dm > 0.0:
                    h_spec = enthalpy[drain_r, drain_z] / max(1e-6, mass[drain_r, drain_z])
                    T_ext = T[drain_r, drain_z]
                    mass[drain_r, drain_z] -= actual_dm
                    enthalpy[drain_r, drain_z] -= actual_dm * h_spec
                    drained_mass_cum += actual_dm
                    drained_this_step += actual_dm
                    cycle_extracted_temp_sum += actual_dm * T_ext
                    if first_drain_time < 0: first_drain_time = time_log[step]
                    last_drain_time = time_log[step]

        did_drain = drained_this_step > 0.0

        # 5. Filling
        filled_this_step = 0.0
        can_fill = False
        if fill_mdot > 0.0:
            if fill_mode == 0: can_fill = True
            elif fill_mode == 1: can_fill = is_heating
            elif fill_mode == 2: can_fill = not is_heating
        if can_fill:
            rho_fill = np.interp(fill_T, T_grid, rho_grid)
            h_fill = np.interp(fill_T, T_grid, H_grid)
            target_mass = fill_mdot * dt
            for i in range(N_r):
                if fill_w[i] > 0.0:
                    target_here = target_mass * fill_w[i]
                    cap_local = V_cell[i, fill_z] * rho_fill * fill_theta_frac[i]
                    current_local = mass[i, fill_z] * fill_theta_frac[i]
                    available_space = cap_local - current_local
                    actual_fill = min(target_here, available_space)
                    if actual_fill > 0.0:
                        mass[i, fill_z] += actual_fill
                        enthalpy[i, fill_z] += actual_fill * h_fill
                        filled_mass_cum += actual_fill
                        filled_this_step += actual_fill
        
        did_fill = filled_this_step > 0.0
        if is_heating and not did_drain and not did_fill:
            idle_time += dt / 3600.0

        drain_log[step] = drained_mass_cum
        fill_log[step] = filled_mass_cum

        for i in range(N_r):
            surf[i] = 0
            for j in range(N_z - 1, -1, -1):
                if mass[i, j] > 1e-6:
                    surf[i] = j
                    break

        # 6. Vertical Flow
        for j in range(0, N_z - 1):
            for i in range(N_r):
                if mass[i, j+1] <= 1e-6: continue
                T_src = T[i, j+1]
                rho_src = np.interp(T_src, T_grid, rho_grid)
                cap_mass = V_cell[i, j] * rho_src
                needed = cap_mass - mass[i, j]
                if needed <= 1e-9: continue
                h_head = (surf[i] - j) * dz
                mu_above = np.interp(T_src, T_grid, mu_grid)
                mobility = (K_settle * h_head) / max(1e-6, mu_above)
                mobility = min(1.0, max(0.01, mobility))
                moved = min(needed * mobility, mass[i, j+1])
                if moved > 1e-9:
                    h_spec = enthalpy[i, j+1] / max(1e-6, mass[i, j+1])
                    mass[i, j] += moved; mass[i, j+1] -= moved
                    enthalpy[i, j] += moved * h_spec
                    enthalpy[i, j+1] -= moved * h_spec

        for j in range(N_z - 2, -1, -1):
            for i in range(N_r):
                if mass[i, j] <= 1e-6: continue
                rho_here = np.interp(T[i, j], T_grid, rho_grid)
                cap_mass = V_cell[i, j] * rho_here
                excess = mass[i, j] - cap_mass
                if excess > 1e-9:
                    h_spec = enthalpy[i, j] / max(1e-6, mass[i, j])
                    mass[i, j] -= excess; mass[i, j+1] += excess
                    enthalpy[i, j] -= excess * h_spec
                    enthalpy[i, j+1] += excess * h_spec

        # 7. Radial Flow
        eff_height = np.zeros(N_r)
        for i in range(N_r):
            area_i = V_cell[i, 0] / dz
            if area_i > 1e-9:
                vol_col = 0.0
                for j in range(N_z):
                    if mass[i, j] > 1e-6:
                        rho_here = np.interp(T[i, j], T_grid, rho_grid)
                        vol_col += mass[i, j] / max(1e-6, rho_here)
                eff_height[i] = vol_col / area_i
        
        for i in range(N_r - 1):
            if eff_height[i] - eff_height[i+1] <= 1e-4: continue
            for j in range(N_z):
                if mass[i, j] <= 1e-6: continue
                j_mid = (j + 0.5) * dz
                p_here = eff_height[i] - j_mid
                p_next = eff_height[i+1] - j_mid
                if p_here <= 0.0 or p_here <= p_next: continue
                local_dh = p_here - max(0.0, p_next)
                if local_dh <= 1e-6: continue
                rho_j = np.interp(T[i, j], T_grid, rho_grid)
                cap = V_cell[i+1, j] * rho_j
                needed = cap - mass[i+1, j]
                if needed <= 1e-9: continue
                mu_j = np.interp(T[i, j], T_grid, mu_grid)
                mobility = (K_settle * local_dh) / max(1e-6, mu_j)
                mobility = min(1.0, max(0.01, mobility))
                moved = min(mass[i, j] * mobility * 0.5, needed)
                if moved > 1e-9:
                    h_spec = enthalpy[i, j] / max(1e-6, mass[i, j])
                    mass[i, j] -= moved; mass[i+1, j] += moved
                    enthalpy[i, j] -= moved * h_spec
                    enthalpy[i+1, j] += moved * h_spec
        
        for i in range(N_r - 2, -1, -1):
            if eff_height[i+1] - eff_height[i] <= 1e-4: continue
            for j in range(N_z):
                if mass[i+1, j] <= 1e-6: continue
                j_mid = (j + 0.5) * dz
                p_here = eff_height[i+1] - j_mid
                p_prev = eff_height[i] - j_mid
                if p_here <= 0.0 or p_here <= p_prev: continue
                local_dh = p_here - max(0.0, p_prev)
                if local_dh <= 1e-6: continue
                rho_j = np.interp(T[i+1, j], T_grid, rho_grid)
                cap = V_cell[i, j] * rho_j
                needed = cap - mass[i, j]
                if needed <= 1e-9: continue
                mu_j = np.interp(T[i+1, j], T_grid, mu_grid)
                mobility = (K_settle * local_dh) / max(1e-6, mu_j)
                mobility = min(1.0, max(0.01, mobility))
                moved = min(mass[i+1, j] * mobility * 0.5, needed)
                if moved > 1e-9:
                    h_spec = enthalpy[i+1, j] / max(1e-6, mass[i+1, j])
                    mass[i+1, j] -= moved; mass[i, j] += moved
                    enthalpy[i+1, j] -= moved * h_spec
                    enthalpy[i, j] += moved * h_spec

        # 8. Wall Loss
        for j in range(N_z):
            if mass[N_r-1, j] > 1e-6:
                area = 2.0 * np.pi * ((N_r - 1) * dr) * dz
                q_loss = U_tank_air * area * (T[N_r-1, j] - T_amb)
                enthalpy[N_r-1, j] -= q_loss * dt

        if not is_heating: status_log[step] = -1
        elif did_drain and did_fill: status_log[step] = 3
        elif did_drain: status_log[step] = 1
        elif did_fill: status_log[step] = 2
        else: status_log[step] = 0

        # Capture Snapshots (Hourly + End of Heating)
        if (step % snapshot_interval_steps == 0) or (is_heating and step == n_steps_heat - 1):
            if snap_count < max_snaps:
                if snap_count > 0 and abs(snapshot_times[snap_count-1] - time_log[step]) < 1e-4:
                    pass
                else:
                    for i in range(N_r):
                        for j in range(N_z):
                            if mass[i, j] <= 1e-6:
                                T_snapshots[snap_count, i, j] = np.nan
                                M_snapshots[snap_count, i, j] = np.nan
                            else:
                                T_snapshots[snap_count, i, j] = T[i, j]
                                M_snapshots[snap_count, i, j] = mass[i, j]
                    snapshot_times[snap_count] = time_log[step]
                    snap_count += 1
            
            if is_heating and step == n_steps_heat - 1:
                liquid_now = 0.0
                for i in range(N_r):
                    for j in range(N_z):
                        if T[i, j] > T_me:
                            liquid_now += mass[i, j]
                
                cycle_remaining_mass = np.sum(mass)
                
                for i in range(N_r):
                    for j in range(N_z):
                        if mass[i, j] > 1e-6:
                            if T[i, j] < T_ms:
                                m_sol += mass[i, j]; T_sol_sum += mass[i, j] * T[i, j]
                            elif T[i, j] <= T_me:
                                m_slu += mass[i, j]; T_slu_sum += mass[i, j] * T[i, j]
                            else:
                                m_liq += mass[i, j]; T_liq_sum += mass[i, j] * T[i, j]

    cycle_extracted = drained_mass_cum - drained_mass_cum_in
    cycle_filled = filled_mass_cum - filled_mass_cum_in
    cycle_idle_hr = idle_time - idle_time_in
    stranded_prev = liquid_now
    avg_T_ext = cycle_extracted_temp_sum / max(1e-6, cycle_extracted)
    
    cyc_sol_T = T_sol_sum / max(1e-6, m_sol)
    cyc_slu_T = T_slu_sum / max(1e-6, m_slu)
    cyc_liq_T = T_liq_sum / max(1e-6, m_liq)
    
    return (mass, enthalpy, T_snapshots, M_snapshots, snapshot_times, snap_count, time_log, drain_log, fill_log, status_log, is_heating_log,
            drained_mass_cum, filled_mass_cum, idle_time, first_drain_time, last_drain_time, stranded_prev,
            cycle_start_mass, cycle_remaining_mass, cycle_extracted, cycle_filled, cycle_idle_hr,
            m_sol, m_slu, m_liq, cyc_sol_T, cyc_slu_T, cyc_liq_T, avg_T_ext)

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
    
    T_start = float(cfg.get("T_start", cfg["T_amb_mid"]))
    rho_start = max(1e-6, np.interp(T_start, mat["T_grid"], mat["rho_grid"]))
    h_spec_start = float(np.interp(T_start, mat["T_grid"], mat["H_grid"]))
    
    M_fluid_eff = float(cfg.get("M_fluid_effective", cfg["M_fluid"]))
    V_total_tank = np.pi * cfg["r_inner"]**2 * tank_height
    V_target = M_fluid_eff / rho_start
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
                mass[i, j] = V_cell * rho_start
                enthalpy[i, j] = mass[i, j] * h_spec_start
            elif j == j_fill_int:
                mass[i, j] = V_cell * rho_start * j_fill_partial
                enthalpy[i, j] = mass[i, j] * h_spec_start
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

    fill_w = np.zeros(N_r)
    fill_theta_frac = np.zeros(N_r)
    fill_flow_m3h = cfg.get("fill_flow_m3h", 0.0)
    fill_T = cfg.get("fill_T", cfg["T_amb_mid"])
    fill_r_idx = cfg.get("fill_r_idx", 0)
    fill_z_idx = cfg.get("fill_z_idx", N_z - 1)
    fill_diameter = cfg.get("fill_diameter", 0.05)
    fill_mode = cfg.get("fill_mode", 0)

    if fill_flow_m3h > 0.0:
        r_fill = fill_r_idx * dr + dr/2.0
        for i in range(N_r):
            r_mid = (i + 0.5) * dr
            if r_mid > 1e-6:
                frac = fill_diameter / (2.0 * np.pi * r_mid)
                fill_theta_frac[i] = min(1.0, frac)
            else:
                fill_theta_frac[i] = 1.0
            
        r_lo_f = max(0.0, r_fill - fill_diameter/2.0)
        r_hi_f = r_fill + fill_diameter/2.0
        total_fw = 0.0
        for i in range(N_r):
            c_r_lo = i * dr; c_r_hi = (i+1) * dr
            ov = max(0.0, min(c_r_hi, r_hi_f) - max(c_r_lo, r_lo_f))
            if ov > 0:
                fill_w[i] = ov
                total_fw += ov
        if total_fw > 0.0:
            fill_w /= total_fw

    drain_r_idx, drain_z_idx = drain_location
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
    snapshot_interval_steps = max(1, int(3600.0 / cfg["dt_sec"]))
    
    fill_mdot = fill_flow_m3h * float(np.interp(fill_T, mat["T_grid"], mat["rho_grid"])) / 3600.0

    drained_mass_cum = 0.0
    filled_mass_cum = 0.0
    idle_time = 0.0
    first_drain_time = -1.0
    last_drain_time = -1.0
    stranded_prev = 0.0
    for i in range(N_r):
        for j in range(N_z):
            if mass[i, j] > 1e-6:
                T_curr = np.interp(enthalpy[i, j] / mass[i, j], mat["H_grid"], mat["T_grid"])
                if T_curr > cfg["T_me"]:
                    stranded_prev += mass[i, j]

    results_2d = {
        "cycle_metrics": [],
        "dt_sec": cfg["dt_sec"],
        "total_hr": (n_cycles * n_steps_heat + (n_cycles - 1) * n_steps_cool) * cfg["dt_sec"] / 3600.0
    }
    
    time_log_all = []
    drain_log_all = []
    fill_log_all = []
    is_heating_log_all = []
    status_log_all = []

    outputs.make_run_dirs(outdir)

    for c in range(n_cycles):
        do_cool = (c < n_cycles - 1)
        
        (mass, enthalpy, T_snaps, M_snaps, snap_times, snap_count, t_log, d_log, f_log, s_log, h_log,
         drained_mass_cum, filled_mass_cum, idle_time, first_drain_time, last_drain_time, stranded_prev,
         start_m, rem_m, ext_kg, fil_kg, idle_h, m_sol, m_slu, m_liq, T_sol, T_slu, T_liq, avg_T_ext) = cycle_kernel(
            mass, enthalpy, N_r, N_z, dr, dz, 
            coil_w, R_coil, A_coil_total, cfg["d_in"], cfg["d_out"], cfg["R_f_out"], 
            cfg["flow_m3h"] * float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"])) / 3600.0,
            drain_r_idx, drain_z_idx, drain_area,
            fill_w, fill_theta_frac, fill_z_idx, fill_mdot, fill_T, fill_mode,
            cfg["T_water_in"], cfg["T_ms"], cfg["T_me"], cfg["Hfus"],
            mat["T_grid"], mat["H_grid"], mat["rho_grid"], mat["cp_sens_grid"], mat["k_grid"], mat["mu_grid"],
            cfg["T_water_arr"], cfg["rho_water_arr"], cfg["cp_water_arr"], cfg["mu_water_arr"], cfg["k_water_arr"],
            cfg["T_tube_arr"], cfg["k_tube_arr"],
            cfg["dt_sec"], n_steps_heat, n_steps_cool, do_cool, n_cycles, cfg["T_amb_mid"], U_tank_air,
            cfg["K_settle"],
            cfg["has_agitator"], cfg["ag_type"], cfg["N_rpm"], cfg["D_imp"], cfg["visc_corr"], cfg["beta_liq"], cfg["dT_natural"],
            drained_mass_cum, filled_mass_cum, idle_time, first_drain_time, last_drain_time, stranded_prev,
            snapshot_interval_steps
        )

        time_log_all.append(t_log)
        drain_log_all.append(d_log)
        fill_log_all.append(f_log)
        is_heating_log_all.append(h_log)
        status_log_all.append(s_log)

        eff = (ext_kg / start_m * 100.0) if start_m > 0 else 0.0
        util = ((cfg["t_window_hr"] - idle_h) / cfg["t_window_hr"] * 100.0) if cfg["t_window_hr"] > 0 else 0.0
        max_possible_fill = fill_flow_m3h * float(np.interp(fill_T, mat["T_grid"], mat["rho_grid"])) * cfg["t_window_hr"]
        fill_util = (fil_kg / max_possible_fill * 100.0) if max_possible_fill > 0 else 0.0

        cycle_metric = {
            "cycle": c + 1, "start_mass": start_m, "remaining_mass": rem_m,
            "extracted_kg": ext_kg, "filled_kg": fil_kg, "extracted_temp_avg": avg_T_ext,
            "efficiency_pct": eff, "utilisation_pct": util, "fill_util_pct": fill_util,
            "heatmaps": [], "massmaps": [], "snapshot_times": snap_times[:snap_count]
        }
        results_2d["cycle_metrics"].append(cycle_metric)

        cdir = outputs.make_cycle_dirs(outdir, c + 1)
        r_grid = np.linspace(0, cfg["r_inner"], N_r)
        z_grid = np.linspace(0, tank_height, N_z)

        for s in range(snap_count):
            hr = snap_times[s]
            h_path = outputs.plot_heat_combined(
                cdir, T_snaps[s], r_grid, z_grid, N_r, N_z, cfg, tank_height, hr, 
                cfg.get("drain_r_idx", 0), cfg.get("drain_angle", 0), 
                cfg.get("fill_r_idx", 0), cfg.get("fill_angle", 0)
            )
            m_path = outputs.plot_mass_combined(
                cdir, M_snaps[s], r_grid, z_grid, N_r, N_z, cfg, tank_height, hr,
                cfg.get("drain_r_idx", 0), cfg.get("drain_angle", 0), 
                cfg.get("fill_r_idx", 0), cfg.get("fill_angle", 0)
            )
            cycle_metric["heatmaps"].append(f"cycles/cycle_{c+1:02d}/{os.path.basename(h_path)}")
            cycle_metric["massmaps"].append(f"cycles/cycle_{c+1:02d}/{os.path.basename(m_path)}")

        outputs.plot_phase_bar(cdir, m_sol, m_slu, m_liq, T_sol, T_slu, T_liq, c)
        outputs.write_cycle_summary(cdir, cycle_metric, c)

        print(f"  Cycle {c+1}/{n_cycles}: Extracted={ext_kg:.1f} kg, Filled={fil_kg:.1f} kg, "
              f"Eff={eff:.1f}%, DrainUtil={util:.1f}%, FillUtil={fill_util:.1f}%")

        results_2d["total_extracted_kg"] = drained_mass_cum
        results_2d["total_filled_kg"] = filled_mass_cum
        results_2d["final_mass"] = float(np.sum(mass))
        results_2d["first_drain"] = first_drain_time
        results_2d["last_drain"] = last_drain_time
        outputs.write_report_html(outdir, cfg, R_coil, n_turns, results_2d)

    time_log_all = np.concatenate(time_log_all)
    drain_log_all = np.concatenate(drain_log_all)
    fill_log_all = np.concatenate(fill_log_all)
    is_heating_log_all = np.concatenate(is_heating_log_all)
    status_log_all = np.concatenate(status_log_all)

    fdir = os.path.join(outdir, "final")
    outputs.plot_flow_curve(fdir, time_log_all, drain_log_all, fill_log_all, is_heating_log_all, cfg["dt_sec"], n_cycles)
    outputs.write_final_summary(fdir, drained_mass_cum, filled_mass_cum, float(np.sum(mass)), first_drain_time, last_drain_time, results_2d["total_hr"])

    results_2d["status_log"] = status_log_all
    results_2d["total_extracted_kg"] = drained_mass_cum
    results_2d["total_filled_kg"] = filled_mass_cum
    results_2d["final_mass"] = float(np.sum(mass))
    results_2d["first_drain"] = first_drain_time
    results_2d["last_drain"] = last_drain_time

    outputs.write_report_html(outdir, cfg, R_coil, n_turns, results_2d)

    with objmode():
        sys.stdout.write("\r  2D Simulation complete. Report generated.\n")
        sys.stdout.flush()

    return results_2d