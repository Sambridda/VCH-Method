# ============================================================
# main.py — The UI, Parallel Driver, and Report Generator
# ============================================================
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import concurrent.futures
import multiprocessing

import engine_1d
import engine_2d

# ============================================================
# WATER REFERENCE TABLE
# ============================================================
WATER_T   = np.array([10., 20., 30., 40., 50., 60., 70., 80., 90., 100.])
WATER_RHO = np.array([999.7, 998.2, 995.7, 992.2, 988.0, 983.2, 977.8, 971.8, 965.3, 958.4])
WATER_CP  = np.array([4192., 4182., 4178., 4179., 4181., 4184., 4187., 4197., 4205., 4216.])
WATER_MU  = np.array([1.308e-3, 1.002e-3, 7.98e-4, 6.53e-4, 5.47e-4, 4.66e-4, 4.04e-4, 3.55e-4, 3.15e-4, 2.82e-4])
WATER_K   = np.array([0.580, 0.598, 0.615, 0.631, 0.644, 0.654, 0.663, 0.670, 0.675, 0.679])

def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "mat"

# ------------------------------------------------------------
# INPUT HELPERS
# ------------------------------------------------------------
def pf(msg, default=None, lo=None, hi=None):
    while True:
        s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        try:
            v = default if (s == "" and default is not None) else float(s)
            if lo is not None and v < lo: print(f"  must be >= {lo}"); continue
            if hi is not None and v > hi: print(f"  must be <= {hi}"); continue
            return v
        except ValueError: print("  invalid number")

def pi_prompt(msg, default=None, lo=None, hi=None):
    while True:
        s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        try:
            v = default if (s == "" and default is not None) else int(s)
            if lo is not None and v < lo: print(f"  must be >= {lo}"); continue
            if hi is not None and v > hi: print(f"  must be <= {hi}"); continue
            return v
        except ValueError: print("  invalid integer")

def pc(msg, choices):
    print(f"\n{msg}")
    for k, label in choices: print(f"  ({k}) {label}")
    keys = [str(k) for k, _ in choices]
    while True:
        s = input("> ").strip()
        if s in keys: return s
        print("  invalid")

def parse_array(msg, default_vals, n):
    default_str = None
    if default_vals is not None:
        if isinstance(default_vals, str): default_str = default_vals
        else: default_str = ", ".join(f"{v:g}" for v in default_vals)
    while True:
        suffix = f" [{default_str}]" if default_str else ""
        s = input(f"{msg}{suffix}: ").strip()
        if not s:
            if default_vals is None: print("  required"); continue
            if isinstance(default_vals, str):
                try: return np.array([float(x) for x in default_vals.split(",")])
                except ValueError: print("  bad default"); continue
            return np.array(default_vals, dtype=float)
        try:
            parts = [x.strip() for x in s.split(",")] if "," in s else s.split()
            vals = [float(x) for x in parts]
            if len(vals) != n: print(f"  expected {n} values, got {len(vals)}"); continue
            return np.array(vals)
        except ValueError as e: print(f"  parse error: {e}")

def _progress(idx, total, t0, prefix="", suffix=""):
    elapsed = time.time() - t0
    frac = idx / total if total > 0 else 0.0
    width = 30; filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled); pct = frac * 100.0
    if idx > 0 and elapsed > 0:
        eta = elapsed * (total - idx) / idx
        eta_s = f"{eta/60:5.1f}m" if eta >= 60 else f"{eta:5.1f}s"
    else: eta_s = "  --  "
    msg = f"  {prefix}[{bar}] {pct:5.1f}% ({idx}/{total})  elapsed {elapsed:6.1f}s  ETA {eta_s}{suffix}"
    sys.stdout.write("\r" + msg.ljust(110)); sys.stdout.flush()
    if idx >= total: sys.stdout.write("\n"); sys.stdout.flush()

# ------------------------------------------------------------
# GATHER INPUTS
# ------------------------------------------------------------
def gather_inputs():
    print("\n" + "="*64)
    print(" TANK COIL DESIGN SIMULATION & 2D DRAINING")
    print("="*64)

    print("\n--- Ambient temperatures degC ---")
    T_amb_low = pf("  Expected cold", 5.0)
    T_amb_mid = pf("  Ambient", 18.0)
    T_amb_hot = pf("  Best case", 25.0)
    
    print("\n--- Operating conditions ---")
    T_water_in = pf("  Coil inlet temperature degC", 70.0); flow_m3h = pf("  Coil volumetric flow m3/h", 1.17)
    beta_liq = pf("  Tank liquid thermal expansion coeff beta (1/K)", 7e-4); dT_natural = pf("  Driving delta-T (K)", 5.0)
    
    print("\n--- Agitator (optional) ---")
    has_ag = 1 if input("  Use an agitator? [y/N]: ").strip().lower() in ("y", "yes") else 0
    if has_ag:
        ag_menu = [("1", "Turbine"), ("2", "Paddle"), ("3", "Anchor"), ("4", "Helical")]
        ag_type = int(pc("  Impeller type:", ag_menu)); N_rpm = pf("  Rotational speed RPM", 60.0)
        D_imp = pf("  Impeller diameter m (0 = tank radius)", 0.0); visc_corr = pf("  Viscosity correction factor", 1.0)
    else: ag_type = 0; N_rpm = 0.0; D_imp = 0.0; visc_corr = 1.0

    def gather_coil_arrays(name, T_low, T_high):
        T_arr = np.linspace(T_low, T_high, 5)
        is_water = "water" in name.lower()
        if is_water:
            rho_d = np.interp(T_arr, WATER_T, WATER_RHO); cp_d = np.interp(T_arr, WATER_T, WATER_CP)
            mu_d = np.interp(T_arr, WATER_T, WATER_MU); k_d = np.interp(T_arr, WATER_T, WATER_K)
        else: rho_d = cp_d = mu_d = k_d = None
        return T_arr, parse_array("    Density kg/m3", rho_d, 5), parse_array("    Specific heat J/(kg*K)", cp_d, 5), parse_array("    Viscosity Pa*s", mu_d, 5), parse_array("    Thermal conductivity W/(m*K)", k_d, 5)

    print("\n--- Coil-side fluid ---")
    fluid_name = input("  Fluid name [water]: ").strip() or "water"
    T_water_low = pf("  Lowest expected coil-fluid temperature degC", T_amb_low)
    (T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr) = gather_coil_arrays(fluid_name, T_water_low, T_water_in)

    print("\n--- Tank-side fluid (PCM) ---")
    tank_name = input("  Name: ").strip() or "oil"
    T_ms = pf("  Melt-start T degC", 31.0); T_me = pf("  Melt-end T degC", 41.0); Hfus = pf("  Latent heat of fusion J/kg", 65000.0)
    
    T_solid_arr = np.linspace(T_amb_low, T_ms, 5)
    print(f"\n  Solid properties at [{', '.join(f'{t:.1f}' for t in T_solid_arr)}] degC:")
    rho_solid_arr = parse_array("    Solid Density kg/m3", [920.0]*5, 5); cp_solid_arr = parse_array("    Solid Cp J/(kg*K)", [1600.0]*5, 5)
    k_solid_arr = parse_array("    Solid k W/(m*K)", [0.180]*5, 5); mu_solid_arr = parse_array("    Solid Viscosity Pa*s", [1e5]*5, 5)

    T_liquid_arr = np.linspace(T_me, T_water_in, 5)
    print(f"\n  Liquid properties at [{', '.join(f'{t:.1f}' for t in T_liquid_arr)}] degC:")
    rho_liq_default = np.array([902.5 - 0.53*T for T in T_liquid_arr]); cp_liq_default = np.array([1750.0 + 2.9*T for T in T_liquid_arr])
    mu_liq_default = np.array([1.85e-6 * np.exp(3080.0/(T+273.15)) for T in T_liquid_arr])
    rho_liquid_arr = parse_array("    Liquid Density kg/m3", rho_liq_default, 5); cp_liquid_arr = parse_array("    Liquid Cp J/(kg*K)", cp_liq_default, 5)
    k_liquid_arr = parse_array("    Liquid k W/(m*K)", [0.85]*5, 5); mu_liquid_arr = parse_array("    Liquid Viscosity Pa*s", mu_liq_default, 5)

    print("\n--- Tank wall ---")
    wall_name = input("  Material name: ").strip() or "steel"
    t_wall = pf("  Wall thickness m", 0.005)
    T_wall_arr = np.linspace(T_amb_low, T_water_in, 5)
    print(f"\n  Wall properties at [{', '.join(f'{t:.1f}' for t in T_wall_arr)}] degC:")
    rho_wall_arr = parse_array("    Density kg/m3", [7850.0]*5, 5); cp_wall_arr = parse_array("    Specific heat J/(kg*K)", [500.0]*5, 5)
    k_wall_arr = parse_array("    Thermal conductivity W/(m*K)", [16.0]*5, 5)

    print("\n--- Coil tube ---")
    coil_mat = input("  Tube material name: ").strip() or "steel"
    d_in = pf("  Inner diameter m", 0.0409); d_out = pf("  Outer diameter m", 0.04826); R_f_out = pf("  Fouling resistance m2*K/W", 0.0004)
    T_tube_arr = np.linspace(T_amb_low, T_water_in, 5)
    print(f"\n  Tube properties at [{', '.join(f'{t:.1f}' for t in T_tube_arr)}] degC:")
    rho_tube_arr = parse_array("    Density kg/m3", [7850.0]*5, 5); cp_tube_arr = parse_array("    Specific heat J/(kg*K)", [500.0]*5, 5)
    k_tube_arr = parse_array("    Thermal conductivity W/(m*K)", [16.0]*5, 5)

    print("\n--- Insulation / environment ---")
    env = pc("  Environment:", [("1", "Indoor (h_air=5)"), ("2", "Outdoor (h_air=10)"), ("3", "Windy (h_air=25)")])
    h_air = {"1": 5.0, "2": 10.0, "3": 25.0}[env]
    t_ins = pf("  Insulation thickness m", 0.0); k_ins = pf("  Insulation k W/(m*K)", 0.04)

    print("\n--- Tank geometry & Setup ---")
    r_inner = pf("  Tank INNER radius m", 1.22)
    L_avail = pf("  AXIAL length available for the coil m", 2.3)
    tank_height = pf("  TOTAL tank height m", 2.8) 
    coil_clearance_bottom = pf("  Clearance at bottom of tank m", 0.2) 
    
    if has_ag and D_imp <= 0.0: D_imp = r_inner

    print("\n--- Make-up Fluid Setup ---")
    has_makeup = 1 if input("  Does the tank get refilled while draining? [y/N]: ").strip().lower() in ("y", "yes") else 0
    makeup_T = pf("  Make-up fluid feed temperature degC", 20.0) if has_makeup else 20.0

    print("\n--- Tank fluid mass ---")
    V_tank = np.pi * r_inner**2 * L_avail
    m_fill = V_tank * float(np.interp(T_amb_low, T_solid_arr, rho_solid_arr))
    s = input(f"  Enter [f] to fill from geometry ({m_fill:.0f} kg), or mass in kg: ").strip().lower()
    M_fluid = m_fill if s in ("f", "") else float(s)

    print("\n--- Simulation Settings ---")
    t_window_hr = pf("  Heating window per phase h", 12.0)
    t_cooldown_hr = pf("  Cooldown window per phase h", 12.0)
    N_cycles = pi_prompt("  Number of cycles (ends on heating)", 1, lo=1, hi=20)
    
    N = pi_prompt("  Radial nodes N (1D)", 100, lo=20, hi=500)
    dt_sec = pf("  Timestep dt s", 1.0, lo=0.1, hi=60.0)
    N_radii = pi_prompt("  Number of radii to explore", 3, lo=2, hi=20)
    n_max_preview = max(1, int(L_avail / d_out))
    n_samples = pi_prompt(f"  Turn samples per radius (1..{n_max_preview})", 10, lo=2, hi=max(2, n_max_preview))

    # Validate Coil Fluid Min Temp
    while True:
        T_eff_min = 0.5 * (T_water_in + T_amb_low)
        if T_eff_min >= T_water_low: break
        print(f"\n  ERROR: Coil-fluid temp would fall below table ({T_water_low}). T_eff_min = {T_eff_min}.")
        if input("  [r] revise or [x] exit: ").lower() == "x": raise SystemExit()
        T_water_low = pf("  New lowest coil-fluid temp degC", T_eff_min)
        (T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr) = gather_coil_arrays(fluid_name, T_water_low, T_water_in)

    return dict(
        fluid_name=fluid_name, T_water_arr=T_water_arr, rho_water_arr=rho_water_arr, cp_water_arr=cp_water_arr, mu_water_arr=mu_water_arr, k_water_arr=k_water_arr, T_water_low=T_water_low,
        tank_name=tank_name, T_ms=T_ms, T_me=T_me, Hfus=Hfus, T_solid_arr=T_solid_arr, rho_solid_arr=rho_solid_arr, cp_solid_arr=cp_solid_arr, k_solid_arr=k_solid_arr, mu_solid_arr=mu_solid_arr,
        T_liquid_arr=T_liquid_arr, rho_liquid_arr=rho_liquid_arr, cp_liquid_arr=cp_liquid_arr, k_liquid_arr=k_liquid_arr, mu_liquid_arr=mu_liquid_arr,
        wall_name=wall_name, t_wall=t_wall, T_wall_arr=T_wall_arr, rho_wall_arr=rho_wall_arr, cp_wall_arr=cp_wall_arr, k_wall_arr=k_wall_arr,
        coil_mat=coil_mat, d_in=d_in, d_out=d_out, R_f_out=R_f_out, T_tube_arr=T_tube_arr, rho_tube_arr=rho_tube_arr, cp_tube_arr=cp_tube_arr, k_tube_arr=k_tube_arr,
        h_air=h_air, t_ins=t_ins, k_ins=k_ins, r_inner=r_inner, L_avail=L_avail, tank_height=tank_height, coil_clearance_bottom=coil_clearance_bottom, M_fluid=M_fluid,
        has_makeup=has_makeup, makeup_T=makeup_T,
        T_water_in=T_water_in, flow_m3h=flow_m3h, T_amb_low=T_amb_low, T_amb_mid=T_amb_mid, T_amb_hot=T_amb_hot,
        t_window_hr=t_window_hr, t_cooldown_hr=t_cooldown_hr, N_cycles=N_cycles, dt_sec=dt_sec, N=N, n_samples=n_samples, N_radii=N_radii,
        has_agitator=has_ag, ag_type=ag_type, N_rpm=N_rpm, D_imp=D_imp, visc_corr=visc_corr, beta_liq=beta_liq, dT_natural=dT_natural
    )

def build_material_tables(cfg):
    T_ms = cfg["T_ms"]; T_me = cfg["T_me"]; Hfus = cfg["Hfus"]; T_amb_low = cfg["T_amb_low"]; T_water_in = cfg["T_water_in"]
    T_min = min(T_amb_low, T_ms) - 20.0; T_max = max(T_water_in, T_me) + 20.0
    n_g = 400; T_grid = np.linspace(T_min, T_max, n_g)
    Ts, Tl = cfg["T_solid_arr"], cfg["T_liquid_arr"]
    rho_s_arr, cp_s_arr, k_s_arr, mu_s_arr = (cfg["rho_solid_arr"], cfg["cp_solid_arr"], cfg["k_solid_arr"], cfg["mu_solid_arr"])
    rho_l_arr, cp_l_arr, k_l_arr, mu_l_arr = (cfg["rho_liquid_arr"], cfg["cp_liquid_arr"], cfg["k_liquid_arr"], cfg["mu_liquid_arr"])
    dT_melt = T_me - T_ms; Hfus_over_dT = Hfus / dT_melt
    rho_grid = np.empty(n_g); cp_sens_grid = np.empty(n_g); cp_eff_grid = np.empty(n_g); k_grid = np.empty(n_g); mu_grid = np.empty(n_g)
    rho_s_end = float(np.interp(T_ms, Ts, rho_s_arr)); rho_l_start = float(np.interp(T_me, Tl, rho_l_arr))
    cp_s_end = float(np.interp(T_ms, Ts, cp_s_arr)); cp_l_start = float(np.interp(T_me, Tl, cp_l_arr))
    k_s_end = float(np.interp(T_ms, Ts, k_s_arr)); k_l_start = float(np.interp(T_me, Tl, k_l_arr))
    mu_s_end = float(np.interp(T_ms, Ts, mu_s_arr)); mu_l_start = float(np.interp(T_me, Tl, mu_l_arr))
    for i in range(n_g):
        T = T_grid[i]
        if T < T_ms:
            rho_grid[i] = np.interp(T, Ts, rho_s_arr); cp_sens_grid[i] = np.interp(T, Ts, cp_s_arr)
            k_grid[i] = np.interp(T, Ts, k_s_arr); mu_grid[i] = np.interp(T, Ts, mu_s_arr); cp_eff_grid[i] = cp_sens_grid[i]
        elif T > T_me:
            rho_grid[i] = np.interp(T, Tl, rho_l_arr); cp_sens_grid[i] = np.interp(T, Tl, cp_l_arr)
            k_grid[i] = np.interp(T, Tl, k_l_arr); mu_grid[i] = np.interp(T, Tl, mu_l_arr); cp_eff_grid[i] = cp_sens_grid[i]
        else:
            f = (T - T_ms) / dT_melt
            rho_grid[i] = rho_s_end*(1-f) + rho_l_start*f; cp_sens_grid[i] = cp_s_end*(1-f) + cp_l_start*f
            k_grid[i] = k_s_end*(1-f) + k_l_start*f; mu_grid[i] = mu_l_start; cp_eff_grid[i] = cp_sens_grid[i] + Hfus_over_dT
    H_grid = np.zeros(n_g)
    for i in range(1, n_g): H_grid[i] = H_grid[i-1] + 0.5*(cp_eff_grid[i-1] + cp_eff_grid[i])*(T_grid[i]-T_grid[i-1])
    return dict(T_grid=T_grid, H_grid=H_grid, rho_grid=rho_grid, cp_sens_grid=cp_sens_grid, k_grid=k_grid, mu_grid=mu_grid)

# ------------------------------------------------------------
# PLOTTING & HTML
# ------------------------------------------------------------
def write_csv(rows, path, cfg):
    if not rows: return
    keys = ["phase","preset","T_amb","start_type","R","R_ratio","n","L_coil","A_coil","liquid_0","slurry_0","solid_0", "liquid","slurry","solid","liquid_change","liquid_T","slurry_T","solid_T", "peak_oil","energy_kWh","peak_power_kW","coil_fluid", "tank_fluid", "wall_material", "coil_material"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            d = dict(r); d["liquid_change"] = r["liquid"] - r.get("liquid_0", 0.0)
            d["coil_fluid"] = cfg["fluid_name"]; d["tank_fluid"] = cfg["tank_name"]; d["wall_material"] = cfg["wall_name"]; d["coil_material"] = cfg["coil_mat"]
            f.write(",".join(str(d.get(k, "")) for k in keys) + "\n")

def plot_cold(rows, T_amb, out_path, t_window_hr, preset_label):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    radii = sorted(set(r["R"] for r in rows)); colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(radii)))
    for j, R in enumerate(radii):
        sub = sorted([r for r in rows if r["R"] == R], key=lambda x: x["n"]); L = [r["L_coil"] for r in sub]; lbl = f"R={R:.2f}"
        ax1.plot(L, [r["liquid"] for r in sub], "-",  color=colors[j], label=f"{lbl} L",   linewidth=1.6)
        ax1.plot(L, [r["slurry"] for r in sub], "--", color=colors[j], label=f"{lbl} S",   linewidth=1.2)
        ax1.plot(L, [r["solid"]  for r in sub], ":",  color=colors[j], label=f"{lbl} Sol", linewidth=1.2)
        ax2.plot(L, [r["liquid_T"] for r in sub], "-",  color=colors[j], label=f"{lbl} liq T")
        ax2.plot(L, [r["slurry_T"] for r in sub], "--", color=colors[j], label=f"{lbl} slu T")
        ax2.plot(L, [r["solid_T"]  for r in sub], ":",  color=colors[j], label=f"{lbl} sol T")
    ax1.set_ylabel("Mass (kg)"); ax1.set_title(f"Cold start — {preset_label} ({T_amb} degC), {t_window_hr:.1f} h heating — L=liquid, S=slurry, Sol=solid")
    ax1.legend(fontsize=7, ncol=3, loc="upper left"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Coil length (m)"); ax2.set_ylabel("Avg T (degC)"); ax2.legend(fontsize=6, ncol=3, loc="best"); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

def _html_table(rows):
    return "<table>\n" + "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "\n</table>"

def _html_prop_arrays(name, T_arr, props):
    h = [f"<h3>{name} — properties at [{', '.join(f'{t:.1f}' for t in T_arr)}] degC</h3>", "<table><tr><th>Property</th>" + "".join(f"<th>{t:.1f} °C</th>" for t in T_arr) + "</tr>"]
    for label, vals, unit in props:
        h.append(f"<tr><td>{label}</td>" + "".join(f"<td>{v:g} {unit}</td>" for v in vals) + "</tr>")
    return "\n".join(h) + "\n</table>"

def write_report_html(outdir, cfg, R_ag, n_ag, pngs, results_2d=None):
    mat = cfg["_mat"]; L_coil = 2.0*np.pi*R_ag*n_ag; A_coil = np.pi*cfg["d_out"]*L_coil
    rho_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["rho_tube_arr"]))
    M_metal = rho_tube_ref * np.pi*0.25*(cfg["d_out"]**2 - cfg["d_in"]**2)*L_coil
    rho_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    V_water = np.pi*0.25*cfg["d_in"]**2*L_coil; M_water = rho_water_ref * V_water

    h = ["<!doctype html><html><head><meta charset='utf-8'><title>Tank Report</title>",
         "<style>body{font-family:sans-serif;max-width:1150px;margin:20px auto;color:#222}",
         "h1{border-bottom:2px solid #333;padding-bottom:6px} h2{margin-top:32px;color:#1a4d7a;border-bottom:1px solid #ddd;padding-bottom:4px}",
         "h3{margin-top:20px;color:#333;font-size:16px;}",
         "table{border-collapse:collapse;margin:8px 0 12px 0;font-size:13px} td,th{border:1px solid #ccc;padding:5px 8px;text-align:right}",
         "th{background:#eef4fa;text-align:center} td:first-child{text-align:left;font-weight:500}",
         ".net-loss{color:#c00;font-weight:bold} .banner{background:#1a4d7a; color:white; padding:12px; border-radius:4px; font-size:16px; margin-bottom:20px}",
         "img{max-width:100%;margin:12px 0;border:1px solid #ddd}</style></head><body>",
         "<h1>Selected Coil — Detailed Report</h1>",
         f"<div class='banner'><b>Materials:</b> Coil Fluid={cfg['fluid_name']} | Tank Fluid={cfg['tank_name']} | Tube={cfg['coil_mat']} | Wall={cfg['wall_name']}</div>",
         f"<p><em>Run: {os.path.basename(outdir)}</em></p>"]

    h.append("<h2>1. Input Parameters</h2>")
    h.append(f"<h3>Coil Fluid: {cfg['fluid_name']}</h3>" + _html_prop_arrays("Coil", cfg["T_water_arr"], [("Density", cfg["rho_water_arr"], "kg/m³"), ("Cp", cfg["cp_water_arr"], "J/kgK"), ("Mu", cfg["mu_water_arr"], "Pa·s"), ("k", cfg["k_water_arr"], "W/mK")]))
    h.append(f"<h3>Tank Fluid: {cfg['tank_name']}</h3>" + _html_prop_arrays("Solid", cfg["T_solid_arr"], [("Density", cfg["rho_solid_arr"], "kg/m³"), ("Cp", cfg["cp_solid_arr"], "J/kgK"), ("k", cfg["k_solid_arr"], "W/mK"), ("Mu", cfg["mu_solid_arr"], "Pa·s")]))
    h.append(_html_prop_arrays("Liquid", cfg["T_liquid_arr"], [("Density", cfg["rho_liquid_arr"], "kg/m³"), ("Cp", cfg["cp_liquid_arr"], "J/kgK"), ("k", cfg["k_liquid_arr"], "W/mK"), ("Mu", cfg["mu_liquid_arr"], "Pa·s")]))
    
    h.append("<h3>Geometry & Operating</h3>" + _html_table([
        ("Inner radius", f"{cfg['r_inner']:.4f} m"), ("Fluid mass", f"{cfg['M_fluid']:.0f} kg"),
        ("Wall", f"{cfg['wall_name']}, t={cfg['t_wall']*1000:.1f} mm"), ("Insulation", f"t={cfg['t_ins']*1000:.0f} mm, U={cfg['h_air']:.1f} W/m²K"),
        ("Agitator", f"Type {cfg['ag_type']} @ {cfg['N_rpm']} RPM" if cfg['has_agitator'] else "None"),
        ("Windows", f"Heating {cfg['t_window_hr']:.2f} h, Cooldown {cfg['t_cooldown_hr']:.2f} h"),
        ("Cycles", f"{cfg['N_cycles']}"),
        ("Ambients", f"Cold: {cfg['T_amb_low']} / Amb: {cfg['T_amb_mid']} / Best: {cfg['T_amb_hot']} °C")
    ]))

    h.append("<h2>2. Selected Coil Geometry</h2>" + _html_table([
        ("Radius R", f"{R_ag:.4f} m"), ("Turns n", f"{n_ag}"), ("Developed length L", f"{L_coil:.2f} m"),
        ("Outer area A", f"{A_coil:.2f} m²"), ("Metal mass", f"{M_metal:.1f} kg"), ("Water volume", f"{V_water*1000:.1f} L")
    ]))

    if results_2d is not None:
        h.append(f"<h2>3. 2D Simulation Results</h2>")
        h.append(f"<p><b>Total Simulation Time:</b> {results_2d['total_hr']:.1f} h<br>")
        h.append(f"<b>Total Liquid Extracted:</b> {results_2d['total_extracted_kg']:.1f} kg</p>")
        
        h.append("<h3>Overall Extraction Curve</h3>")
        h.append(f"<img src='{os.path.basename(results_2d['curve_path'])}' alt='curve'>")
        
        for idx, m in enumerate(results_2d["cycle_metrics"]):
            h.append(f"<h3>Cycle {m['cycle']} Metrics</h3>")
            h.append("<table>")
            h.append(f"<tr><td>Liquid Produced</td><td>{m['produced_kg']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Liquid Extracted</td><td>{m['extracted_kg']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Stranded in Tank</td><td>{m['stranded_kg']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Extraction Efficiency</td><td>{m['efficiency_pct']:.1f} %</td></tr>")
            h.append(f"<tr><td>Idle Rate</td><td>{m['idle_pct']:.1f} %</td></tr>")
            h.append("</table>")
            
            h.append(f"<h3>Cycle {m['cycle']} Heatmap (End of Heating)</h3>")
            h.append(f"<img src='{os.path.basename(results_2d['heatmap_paths'][idx])}' alt='heatmap_{idx}'>")

            h.append(f"<h3>Cycle {m['cycle']} Mass Map (End of Heating)</h3>")
            h.append(f"<img src='{os.path.basename(results_2d['massmap_paths'][idx])}' alt='massmap_{idx}'>")

            h.append(f"<h3>Cycle {m['cycle']} Phase Distribution</h3>")
            h.append(f"<img src='{os.path.basename(results_2d['phasebar_paths'][idx])}' alt='phasebar_{idx}'>")

    h.append("</body></html>")

    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(h))

# ------------------------------------------------------------
# MAIN ORCHESTRATOR
# ------------------------------------------------------------
def main():
    cfg = gather_inputs()
    mat = build_material_tables(cfg)
    cfg["_mat"] = mat

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("out", f"run_{ts}_{sanitize_filename(cfg['fluid_name'])}_{sanitize_filename(cfg['tank_name'])}")
    os.makedirs(outdir, exist_ok=True)
    print(f"\nOutput: {outdir}")

    geom, n_max, radii, turns = engine_1d.build_geometry(cfg)
    
    # Updated Presets based on user request
    presets = [("expected_cold", cfg["T_amb_low"]), ("ambient", cfg["T_amb_mid"]), ("best_case", cfg["T_amb_hot"])]

    print(f"\n--- Phase A: cold-start sweep ({len(radii)} radii x {len(turns)} turns x 3 ambients) ---")
    print("Warming up Numba JIT compiler...")
    _ = engine_1d.run_one(cfg, mat, radii[0], turns[0], cfg["T_amb_low"], None, 1.0/3600.0, "heat")
    
    total_A = len(radii) * len(turns) * len(presets)
    done_A = 0; t_A = time.time()
    tasks = []
    for R in radii:
        for pname, Tamb in presets:
            for n in turns:
                tasks.append((cfg, mat, R, n, Tamb, pname, cfg["t_window_hr"]))
            
    cold_rows_all = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(engine_1d.run_sweep_task, task) for task in tasks]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result(); cold_rows_all.append(r); done_A += 1
            _progress(done_A, total_A, t_A, prefix="Phase A (multi-core) ")

    for pname, Tamb in presets:
        rows = [r for r in cold_rows_all if r["preset"] == pname]
        plot_cold(rows, Tamb, os.path.join(outdir, f"cold_{pname}.png"), cfg["t_window_hr"], pname.replace("_", " ").title())
        write_csv(rows, os.path.join(outdir, f"cold_{pname}.csv"), cfg)

    print("\n--- Phase B: pick agreed dimension ---")
    menu = []
    for ri, R in enumerate(radii, 1):
        for g in sorted([g for g in geom if g["R"] == R], key=lambda x: x["n"]):
            menu.append({"idx": len(menu)+1, "R_idx": ri, "R": g["R"], "n": g["n"], "L": g["L"], "R_ratio": g["R_ratio"]})
    lookup = {}
    for r in cold_rows_all: lookup.setdefault((round(r["R"], 6), r["n"]), {})[r["preset"]] = r["liquid"]
    menu_path = os.path.join(outdir, "picker_menu.txt")
    with open(menu_path, "w", encoding="utf-8") as f:
        f.write(f"{'idx':>4}  {'R(m)':>7}  {'n':>3}  {'L(m)':>7}  {'liq_cold':>9}  {'liq_amb':>9}  {'liq_best':>9}\n")
        for m in menu:
            vals = lookup.get((round(m["R"], 6), m["n"]), {})
            f.write(f"{m['idx']:>4}  {m['R']:>7.3f}  {m['n']:>3}  {m['L']:>7.1f}  {vals.get('expected_cold',0):>9.0f}  {vals.get('ambient',0):>9.0f}  {vals.get('best_case',0):>9.0f}\n")

    print(f"  Phase A complete — {len(menu)} options written to: {menu_path}")
    chosen = None
    while chosen is None:
        s = input(f"  idx (1-{len(menu)}): ").strip()
        try:
            idx = int(s)
            if 1 <= idx <= len(menu): chosen = menu[idx-1]
            else: print(f"  out of range (1-{len(menu)})")
        except ValueError: print("  enter a number")
    R_ag = chosen["R"]; n_ag = chosen["n"]; L_ag = chosen["L"]
    print(f"  Agreed: R={R_ag:.3f} m  n={n_ag}  L={L_ag:.1f} m")

    # --- PHASE C: 2D SIMULATION SETUP ---
    print("\n" + "="*70)
    print(" PHASE C: 2D SIMULATION SETUP")
    print("="*70)
    
    simulate_drain = input(" Do you want to simulate continuous draining? [y/N]: ").strip().lower() in ("y", "yes")
    
    if simulate_drain:
        drain_diameter = pf(" Enter drain valve diameter (m)", default=0.05, lo=0.001, hi=1.0)
        max_idx = 49 # 50x50 grid
        drain_r_idx = pi_prompt(f" Drain radial index (0=center, {max_idx}=wall)", default=0, lo=0, hi=max_idx)
        drain_z_idx = pi_prompt(f" Drain axial index (0=bottom, {max_idx}=top)", default=0, lo=0, hi=max_idx)
    else:
        drain_diameter = 0.0
        drain_r_idx = 0
        drain_z_idx = 0

    # --- PHASE D: THE 2D SIMULATION ---
    print("\n" + "="*70)
    print(" PHASE D: 2D SIMULATION")
    print("="*70)
    
    results_2d = engine_2d.run_2d_simulation(
        cfg, mat, R_ag, n_ag, None, drain_diameter, (drain_r_idx, drain_z_idx), outdir
    )
    
    pngs = []
    pngs.append(results_2d["curve_path"])
    pngs.extend(results_2d["heatmap_paths"])
    pngs.extend(results_2d["massmap_paths"])
    pngs.extend(results_2d["phasebar_paths"])
    
    print("\n--- 2D SIMULATION RESULTS ---")
    for m in results_2d["cycle_metrics"]:
        print(f"  Cycle {m['cycle']}: Extracted={m['extracted_kg']:.1f} kg, Efficiency={m['efficiency_pct']:.1f}%, Idle={m['idle_pct']:.1f}%")

    write_report_html(outdir, cfg, R_ag, n_ag, pngs, results_2d)

    print("\n" + "="*70 + "\n DONE\n" + "="*70)
    print(f"  {outdir}/index.html")

if __name__ == "__main__":
    main()