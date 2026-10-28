# ui.py — Console input and output
import numpy as np
import sys
import time
import materials

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

def progress(idx, total, t0, prefix="", suffix=""):
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

def gather_initial_inputs():
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

    avail_mats = materials.list_available()
    
    def pick_material(title, default_idx=0):
        print(f"\n--- {title} ---")
        for i, m in enumerate(avail_mats):
            print(f"  ({i+1}) {m}")
        while True:
            s = input(f"Select material [1-{len(avail_mats)}] [{default_idx+1}]: ").strip()
            if s == "": return avail_mats[default_idx]
            try:
                idx = int(s) - 1
                if 0 <= idx < len(avail_mats): return avail_mats[idx]
            except ValueError: pass
            print("  invalid selection")

    coil_mat_name = pick_material("Coil-side Fluid")
    tank_mat_name = pick_material("Tank-side Fluid (PCM)")
    wall_mat_name = pick_material("Tank Wall Material")
    tube_mat_name = pick_material("Coil Tube Material")
    
    thresh_liq, thresh_visc = materials.load_settings()

    print("\n--- Tank geometry & Setup ---")
    r_inner = pf("  Tank INNER radius m", 1.22)
    L_avail = pf("  AXIAL length available for the coil m", 2.3)
    tank_height = pf("  TOTAL tank height m", 2.8) 
    coil_clearance_bottom = pf("  Clearance at bottom of tank m", 0.2) 

    print("\n--- Tank fluid mass ---")
    mat_tank = materials.load(tank_mat_name)
    V_tank = np.pi * r_inner**2 * tank_height
    m_fill = V_tank * float(np.interp(T_amb_low, mat_tank["T_grid"], mat_tank["rho_grid"]))
    print(f"  Maximum capacity (to tank rim): {m_fill:.0f} kg")
    s = input(f"  Enter [f] to fill to max ({m_fill:.0f} kg), or mass in kg: ").strip().lower()
    if s in ("f", ""):
        M_fluid = m_fill
    else:
        M_fluid = float(s)
        if M_fluid > m_fill:
            print(f"  WARNING: {M_fluid:.0f} kg exceeds tank capacity. Capping at {m_fill:.0f} kg.")
            M_fluid = m_fill

    print("\n--- Simulation Settings ---")
    t_window_hr = pf("  Heating window per phase h", 12.0)
    t_cooldown_hr = pf("  Cooldown window per phase h", 12.0)
    N_cycles = pi_prompt("  Number of cycles (ends on heating)", 1, lo=1, hi=20)
    
    print("\n  Settling Coefficient (K_settle):")
    print("    Scales how fast the fluid settles vertically under gravity.")
    K_settle = pf("  Settling Coefficient K_settle", 1.0, lo=0.001, hi=100.0)

    N = pi_prompt("  Radial nodes N (1D)", 100, lo=20, hi=500)
    dt_sec = pf("  Timestep dt s", 1.0, lo=0.1, hi=60.0)
    N_radii = pi_prompt("  Number of radii to explore", 3, lo=2, hi=20)
    
    n_max_preview = max(1, int(L_avail / 0.05))
    n_samples = pi_prompt(f"  Turn samples per radius (1..{n_max_preview})", 10, lo=2, hi=max(2, n_max_preview))

    return dict(
        fluid_name=coil_mat_name, tank_name=tank_mat_name, wall_name=wall_mat_name, coil_mat=tube_mat_name,
        T_water_in=T_water_in, flow_m3h=flow_m3h, T_amb_low=T_amb_low, T_amb_mid=T_amb_mid, T_amb_hot=T_amb_hot,
        t_window_hr=t_window_hr, t_cooldown_hr=t_cooldown_hr, N_cycles=N_cycles, dt_sec=dt_sec, N=N, n_samples=n_samples, N_radii=N_radii,
        beta_liq=beta_liq, dT_natural=dT_natural, K_settle=K_settle,
        r_inner=r_inner, L_avail=L_avail, tank_height=tank_height, coil_clearance_bottom=coil_clearance_bottom, M_fluid=M_fluid,
        thresh_liq=thresh_liq, thresh_visc=thresh_visc,
        d_in=0.0409, d_out=0.04826, R_f_out=0.0004, h_air=5.0, t_wall=0.005, t_ins=0.0, k_ins=0.04
    )

def gather_2d_inputs(cfg, mat_tank):
    print("\n" + "="*70)
    print(" PHASE C: 2D SIMULATION SETUP")
    print("="*70)

    print("\n  The 2D simulation needs to know what the tank contains at t=0.")
    print(f"  Environment (for heat loss) will stay at T_ambient = {cfg['T_amb_mid']:.1f} °C.")
    
    T_ms = mat_tank.get("T_ms", np.nan)
    T_me = mat_tank.get("T_me", np.nan)
    if not np.isnan(T_ms) and not np.isnan(T_me):
        print(f"  Melt band: {T_ms:.1f} → {T_me:.1f} °C")
    else:
        print("  Melt band: None (Single-phase fluid)")

    T_start = pf("  Starting tank fluid temperature °C", cfg["T_amb_mid"])
    rho_start = float(np.interp(T_start, mat_tank["T_grid"], mat_tank["rho_grid"]))
    V_tank_full = np.pi * cfg["r_inner"]**2 * cfg["tank_height"]
    V_target = cfg["M_fluid"] / max(1e-6, rho_start)

    if V_target > V_tank_full:
        actual_kg = V_tank_full * rho_start
        print(f"  ⚠ Tank will be filled to {actual_kg:.0f} kg (overflow excluded).")
        cfg["M_fluid_effective"] = actual_kg
    else:
        print(f"  ✓ {cfg['M_fluid']:.0f} kg fills {V_target/V_tank_full*100:.1f}% of the tank.")
        cfg["M_fluid_effective"] = cfg["M_fluid"]

    cfg["T_start"] = T_start

    # Dynamically set the max index based on the current 2D grid resolution.
    # In engine_2d.py, res = 5, N_r = 10*res = 50. So indices go from 0 to 49.
    max_idx = 49 

    print("\n--- 2D Drain Setup ---")
    cfg["simulate_drain"] = input("  Do you want to simulate continuous draining? [y/N]: ").strip().lower() in ("y", "yes")
    if cfg["simulate_drain"]:
        cfg["drain_diameter"] = pf("    Drain valve diameter (m)", default=0.05, lo=0.001, hi=1.0)
        cfg["drain_r_idx"] = pi_prompt(f"    Drain radial index (0=center, {max_idx}=wall)", default=0, lo=0, hi=max_idx)
        cfg["drain_angle"] = pf("    Drain angular position (0-360)", default=0.0, lo=0.0, hi=360.0) if cfg["drain_r_idx"] > 0 else 0.0
        cfg["drain_z_idx"] = pi_prompt(f"    Drain axial index (0=bottom, {max_idx}=top)", default=0, lo=0, hi=max_idx)
    else:
        cfg.update({"drain_diameter": 0.0, "drain_r_idx": 0, "drain_z_idx": 0, "drain_angle": 0.0})

    print("\n--- 2D Fill Setup ---")
    cfg["simulate_fill"] = input("  Do you want to simulate continuous filling? [y/N]: ").strip().lower() in ("y", "yes")
    if cfg["simulate_fill"]:
        cfg["fill_diameter"] = pf("    Fill nozzle diameter (m)", default=0.05, lo=0.001, hi=1.0)
        cfg["fill_flow_m3h"] = pf("    Fill volumetric flow rate (m3/h)", default=1.0, lo=0.01, hi=100.0)
        cfg["fill_T"] = pf("    Fill fluid temperature degC", default=cfg["T_amb_mid"])
        cfg["fill_r_idx"] = pi_prompt(f"    Fill radial index (0=center, {max_idx}=wall)", default=0, lo=0, hi=max_idx)
        cfg["fill_angle"] = pf("    Fill angular position (0-360)", default=0.0, lo=0.0, hi=360.0) if cfg["fill_r_idx"] > 0 else 0.0
        cfg["fill_z_idx"] = pi_prompt(f"    Fill axial index (0=bottom, {max_idx}=top)", default=max_idx, lo=0, hi=max_idx)
        cfg["fill_mode"] = int(pc("    Fill behaviour:", [("0", "Constant"), ("1", "Heating only"), ("2", "Cooling only")]))
    else:
        cfg.update({"fill_diameter": 0.0, "fill_flow_m3h": 0.0, "fill_T": cfg["T_amb_mid"], "fill_r_idx": 0, "fill_z_idx": max_idx, "fill_mode": 0, "fill_angle": 0.0})

    return cfg

def pick_coil(menu):
    print("\n--- Phase B: pick agreed dimension ---")
    chosen = None
    while chosen is None:
        s = input(f"  idx (1-{len(menu)}): ").strip()
        try:
            idx = int(s)
            if 1 <= idx <= len(menu): chosen = menu[idx-1]
            else: print(f"  out of range (1-{len(menu)})")
        except ValueError: print("  enter a number")
    print(f"  Agreed: R={chosen['R']:.3f} m  n={chosen['n']}  L={chosen['L']:.1f} m")
    return chosen["R"], chosen["n"]

def print_cycle(c):
    print(f"  Cycle {c['cycle']}/{c['total_cycles']}: Extracted={c['extracted_kg']:.1f} kg, "
          f"Filled={c['filled_kg']:.1f} kg, Eff={c['efficiency_pct']:.1f}%, "
          f"DrainUtil={c['utilisation_pct']:.1f}%, FillUtil={c['fill_util_pct']:.1f}%")

def print_final(results_2d, outdir):
    print("\n" + "="*70 + "\n DONE\n" + "="*70)
    print(f"  {outdir}/index.html")