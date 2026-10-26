# ============================================================
# engine_1d.py — 1D Radial Physics Engine (Pure Math)
# ============================================================
import numpy as np
from numba import njit

@njit(cache=True)
def h_internal(m_dot, d_in, d_coil_actual, rho, cp, mu, k):
    A = np.pi * (d_in/2.0)**2
    v = m_dot / (rho * A)
    Re = rho * v * d_in / mu
    Pr = mu * cp / k
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
        Re_ia = rho * N_rps * (D_imp**2) / mu
        Pr = mu * cp / k
        if ag_type == 1: C, a = 0.60, 0.67
        elif ag_type == 2: C, a = 0.54, 0.67
        elif ag_type == 3: C, a = 0.40, 0.50
        else: C, a = 0.35, 0.50
        Nu = C * (Re_ia ** a) * (Pr ** (1.0/3.0)) * visc_corr
        return Nu * k / d_out
    else:
        nu = mu / rho; alpha = k / (rho * cp); Pr = nu / alpha
        d_char = d_out
        Ra = 9.81 * beta * dT_natural * (d_char**3) / (nu * alpha)
        Nu = (0.60 + (0.387 * Ra**(1.0/6.0)) / (1.0 + (0.559/Pr)**(9.0/16.0))**(8.0/27.0))**2
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
def transient_kernel(H, T_steel, N, dr, dt_inv_m, A_face, inj_w,
                     T_ms, T_me,
                     T_grid, H_grid, rho_grid, cp_sens_grid, k_grid, mu_grid,
                     T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
                     m_dot, T_water_in,
                     d_in, d_out, R_f_out, A_coil, d_coil_actual,
                     T_tube_arr, k_tube_arr,
                     M_steel,
                     T_wall_arr, k_wall_arr, cp_wall_arr,
                     A_tank_outer, t_wall, t_ins, k_ins, h_air,
                     T_amb, dt, n_steps,
                     has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural):
    peak_oil = -1e9; Q_total = 0.0; peak_Q = 0.0
    T = np.empty(N); k_arr = np.empty(N)

    for step in range(n_steps):
        for i in range(N):
            T[i] = np.interp(H[i], H_grid, T_grid)
            if T[i] > peak_oil: peak_oil = T[i]
        for i in range(N):
            k_arr[i] = np.interp(T[i], T_grid, k_grid)

        step_Q = 0.0
        for i in range(N):
            if inj_w[i] > 0.0:
                Q_local = calc_Q(m_dot, T[i], T_water_in, d_in, d_out, R_f_out, A_coil, d_coil_actual,
                                 T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr,
                                 T_tube_arr, k_tube_arr,
                                 T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms,
                                 has_agitator, ag_type, N_rpm, D_imp, visc_corr, beta, dT_natural)
                H[i] += Q_local * inj_w[i] * dt_inv_m[i]
                Q_total += Q_local * inj_w[i] * dt
                step_Q += Q_local * inj_w[i]
        if step_Q > peak_Q: peak_Q = step_Q

        for i in range(N-1):
            kavg = 0.5*(k_arr[i] + k_arr[i+1])
            Qc = kavg * A_face[i] * (T[i] - T[i+1]) / dr
            H[i]   -= Qc * dt_inv_m[i]
            H[i+1] += Qc * dt_inv_m[i+1]

        Q_os = k_arr[N-1] * A_tank_outer * (T[N-1] - T_steel) / (dr*0.5)
        H[N-1] -= Q_os * dt_inv_m[N-1]

        k_wall_now  = np.interp(T_steel, T_wall_arr, k_wall_arr)
        cp_wall_now = np.interp(T_steel, T_wall_arr, cp_wall_arr)
        R_conv = 1.0 / h_air
        R_wall_half = (t_wall/2.0) / k_wall_now
        R_ins = (t_ins/k_ins) if t_ins > 0.0 else 0.0
        U_air = 1.0 / (R_wall_half + R_ins + R_conv)
        Q_sa = U_air * A_tank_outer * (T_steel - T_amb)
        T_steel += (Q_os - Q_sa) * dt / (M_steel * cp_wall_now)

    for i in range(N):
        Ti = np.interp(H[i], H_grid, T_grid)
        if Ti > peak_oil: peak_oil = Ti
    return H, T_steel, peak_oil, Q_total, peak_Q

@njit(cache=True)
def cooldown_kernel(H, T_steel, N, dr, dt_inv_m_eff, A_face,
                    T_grid, H_grid, k_grid, M_steel,
                    T_wall_arr, k_wall_arr, cp_wall_arr,
                    A_tank_outer, t_wall, t_ins, k_ins, h_air,
                    T_amb, dt, n_steps):
    T = np.empty(N); k_arr = np.empty(N)
    for step in range(n_steps):
        for i in range(N):
            T[i] = np.interp(H[i], H_grid, T_grid)
        for i in range(N):
            k_arr[i] = np.interp(T[i], T_grid, k_grid)
        for i in range(N-1):
            kavg = 0.5*(k_arr[i] + k_arr[i+1])
            Qc = kavg * A_face[i] * (T[i] - T[i+1]) / dr
            H[i]   -= Qc * dt_inv_m_eff[i]
            H[i+1] += Qc * dt_inv_m_eff[i+1]
        Q_os = k_arr[N-1] * A_tank_outer * (T[N-1] - T_steel) / (dr*0.5)
        H[N-1] -= Q_os * dt_inv_m_eff[N-1]

        k_wall_now  = np.interp(T_steel, T_wall_arr, k_wall_arr)
        cp_wall_now = np.interp(T_steel, T_wall_arr, cp_wall_arr)
        R_conv = 1.0 / h_air
        R_wall_half = (t_wall/2.0) / k_wall_now
        R_ins = (t_ins/k_ins) if t_ins > 0.0 else 0.0
        U_air = 1.0 / (R_wall_half + R_ins + R_conv)
        Q_sa = U_air * A_tank_outer * (T_steel - T_amb)
        T_steel += (Q_os - Q_sa) * dt / (M_steel * cp_wall_now)
    return H, T_steel

# ============================================================
# PURE PYTHON HELPERS
# ============================================================
def whole_turns_logspace(n_min, n_max, npts):
    if n_max <= n_min: return np.array([n_min])
    if n_max - n_min + 1 <= npts: return np.arange(n_min, n_max + 1)
    raw = np.logspace(np.log10(n_min), np.log10(n_max), npts)
    turns = []
    for r in raw:
        v = max(n_min, min(n_max, int(round(r))))
        while v in turns and v < n_max: v += 1
        while v in turns and v > n_min: v -= 1
        turns.append(v)
    return np.array(sorted(set(turns)))

def build_geometry(cfg):
    r = cfg["r_inner"]; L = cfg["L_avail"]; d_out = cfg["d_out"]
    radii = np.linspace(0.33*r, 0.75*r, cfg["N_radii"])
    n_max = max(1, int(L/d_out))
    turns = whole_turns_logspace(1, n_max, cfg["n_samples"])
    rows = []
    for R in radii:
        for n in turns:
            Lc = 2.0*np.pi*R*n
            rows.append(dict(R=float(R), n=int(n), L=float(Lc), A=np.pi*d_out*Lc, R_ratio=float(R/r)))
    return rows, n_max, radii, turns

def prepare_mesh(cfg, mat, T_ref):
    N = cfg["N"]; r = cfg["r_inner"]; L = cfg["L_avail"]; dr = r/N
    idx = np.arange(N)
    r_in = idx*dr; r_out = (idx+1)*dr; r_mid = 0.5*(r_in + r_out)
    A_face = 2.0*np.pi*r_mid[:-1]*L
    rho_ref = float(np.interp(T_ref, mat["T_grid"], mat["rho_grid"]))
    m_shell = np.pi*(r_out**2 - r_in**2)*L*rho_ref
    return dr, r_in, r_out, r_mid, A_face, m_shell

def injection_weights(R_coil, d_out, N, dr):
    band_lo = R_coil - d_out/2.0; band_hi = R_coil + d_out/2.0
    w_in  = np.zeros(N); w_out = np.zeros(N)
    for i in range(N):
        r_in = i*dr; r_out = (i+1)*dr
        lo = max(r_in, band_lo); hi = min(r_out, band_hi)
        if hi <= lo: continue
        overlap = hi - lo
        if r_out <= R_coil: w_in[i] += overlap
        elif r_in >= R_coil: w_out[i] += overlap
        else:
            f_in = (R_coil - r_in)/(r_out - r_in)
            w_in[i]  += overlap * f_in
            w_out[i] += overlap * (1.0 - f_in)
    si = w_in.sum(); so = w_out.sum()
    if si > 0: w_in  /= si
    if so > 0: w_out /= so
    area_in = R_coil - d_out / 2.0; area_out = R_coil + d_out / 2.0
    frac_out = area_out / (area_in + area_out); frac_in = 1.0 - frac_out
    return frac_in * w_in + frac_out * w_out

def steel_mass(cfg, T_ref):
    r = cfg["r_inner"]; t = cfg["t_wall"]; L = cfg["L_avail"]
    V = np.pi*((r+t)**2 - r**2)*L
    rho_ref = float(np.interp(T_ref, cfg["T_wall_arr"], cfg["rho_wall_arr"]))
    return rho_ref * V

def tank_outer_area(cfg):
    Ro = cfg["r_inner"] + cfg["t_wall"]
    return 2.0*np.pi*Ro*cfg["L_avail"] + 2.0*np.pi*Ro**2

def classify(cfg, mat, H, m_shell):
    N = cfg["N"]
    liq = slu = sol = 0.0; liqT = sluT = solT = 0.0
    for i in range(N):
        T = float(np.interp(H[i], mat["H_grid"], mat["T_grid"]))
        m = m_shell[i]
        if T > cfg["T_me"]:    liq += m; liqT += m*T
        elif T >= cfg["T_ms"]: slu += m; sluT += m*T
        else:                   sol += m; solT += m*T
    return dict(liquid=liq, slurry=slu, solid=sol,
                liquid_T=liqT/liq if liq > 0 else 0.0,
                slurry_T=sluT/slu if slu > 0 else 0.0,
                solid_T =solT/sol if sol > 0 else 0.0)

def run_one(cfg, mat, R_coil, n_turns, T_amb, T_initial, t_hr, mode):
    N = cfg["N"]
    T_ref_mesh = T_amb if T_initial is None else float(T_initial["T_steel"])
    dr, r_in, r_out, r_mid, A_face, m_shell = prepare_mesh(cfg, mat, T_ref_mesh)
    inj_w = injection_weights(R_coil, cfg["d_out"], N, dr)
    d_coil_actual = 2.0 * R_coil

    if T_initial is None:
        H = np.full(N, float(np.interp(T_amb, mat["T_grid"], mat["H_grid"])))
        T_steel = T_amb
    else:
        H = np.array(T_initial["H"]); T_steel = float(T_initial["T_steel"])

    M_steel = steel_mass(cfg, 0.5*(cfg["T_amb_low"] + cfg["T_water_in"]))
    A_outer = tank_outer_area(cfg)
    dt = cfg["dt_sec"]
    n_steps = int(round(t_hr * 3600.0 / dt))

    if mode == "cool":
        L_coil = 2.0*np.pi*R_coil*n_turns
        V_metal = np.pi*0.25*(cfg["d_out"]**2 - cfg["d_in"]**2)*L_coil
        V_water = np.pi*0.25*cfg["d_in"]**2*L_coil
        rho_tube_ref  = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"],  cfg["rho_tube_arr"]))
        cp_tube_ref   = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"],  cfg["cp_tube_arr"]))
        rho_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
        cp_water_ref  = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["cp_water_arr"]))
        M_coil_metal = rho_tube_ref * V_metal
        M_coil_water = rho_water_ref * V_water
        cp_oil_ref = float(np.interp(cfg["T_water_in"], mat["T_grid"], mat["cp_sens_grid"]))
        extra_kg = (M_coil_metal*cp_tube_ref + M_coil_water*cp_water_ref) / cp_oil_ref
        m_eff = m_shell + extra_kg*inj_w
        dt_inv_m_eff = dt / m_eff

        H_out, T_steel = cooldown_kernel(
            H, T_steel, N, dr, dt_inv_m_eff, A_face,
            mat["T_grid"], mat["H_grid"], mat["k_grid"], M_steel,
            cfg["T_wall_arr"], cfg["k_wall_arr"], cfg["cp_wall_arr"],
            A_outer, cfg["t_wall"], cfg["t_ins"], cfg["k_ins"], cfg["h_air"],
            T_amb, dt, n_steps)
        peak_oil = 0.0; Q_total = 0.0; peak_Q = 0.0
    else:
        rho_inlet = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
        m_dot = cfg["flow_m3h"] * rho_inlet / 3600.0
        A_coil = np.pi*cfg["d_out"]*(2.0*np.pi*R_coil*n_turns)
        dt_inv_m = dt / m_shell

        H_out, T_steel, peak_oil, Q_total, peak_Q = transient_kernel(
            H, T_steel, N, dr, dt_inv_m, A_face, inj_w,
            cfg["T_ms"], cfg["T_me"],
            mat["T_grid"], mat["H_grid"], mat["rho_grid"], mat["cp_sens_grid"], mat["k_grid"], mat["mu_grid"],
            cfg["T_water_arr"], cfg["rho_water_arr"], cfg["cp_water_arr"], cfg["mu_water_arr"], cfg["k_water_arr"],
            m_dot, cfg["T_water_in"],
            cfg["d_in"], cfg["d_out"], cfg["R_f_out"], A_coil, d_coil_actual,
            cfg["T_tube_arr"], cfg["k_tube_arr"], M_steel,
            cfg["T_wall_arr"], cfg["k_wall_arr"], cfg["cp_wall_arr"],
            A_outer, cfg["t_wall"], cfg["t_ins"], cfg["k_ins"], cfg["h_air"],
            T_amb, dt, n_steps,
            cfg["has_agitator"], cfg["ag_type"], cfg["N_rpm"], cfg["D_imp"], cfg["visc_corr"],
            cfg["beta_liq"], cfg["dT_natural"])

    st = classify(cfg, mat, H_out, m_shell)
    st0 = classify(cfg, mat, np.array(T_initial["H"]), m_shell) if T_initial else dict(liquid=0.0, slurry=0.0, solid=0.0, liquid_T=0.0, slurry_T=0.0, solid_T=0.0)

    return dict(
        R=R_coil, n=int(n_turns), L_coil=2.0*np.pi*R_coil*n_turns,
        A_coil=np.pi*cfg["d_out"]*(2.0*np.pi*R_coil*n_turns),
        liquid_0=st0["liquid"], slurry_0=st0["slurry"], solid_0=st0["solid"],
        liquid=st["liquid"], slurry=st["slurry"], solid=st["solid"],
        liquid_T=st["liquid_T"], slurry_T=st["slurry_T"], solid_T=st["solid_T"],
        peak_oil=peak_oil, energy_kWh=Q_total/3.6e6, peak_power_kW=peak_Q/1000.0,
        H_final=H_out.tolist(), T_steel=float(T_steel),
    )

def run_sweep_task(args):
    cfg, mat, R, n, Tamb, pname, t_window_hr = args
    r = run_one(cfg, mat, R, n, Tamb, None, t_window_hr, "heat")
    r["T_amb"] = Tamb
    r["preset"] = pname
    r["start_type"] = "cold"
    r["R_ratio"] = R / cfg["r_inner"]
    return r