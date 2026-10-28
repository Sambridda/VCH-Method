# outputs.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

def make_run_dirs(outdir):
    paths = {
        "cycles": os.path.join(outdir, "cycles"),
        "final": os.path.join(outdir, "final"),
    }
    os.makedirs(paths["cycles"], exist_ok=True)
    os.makedirs(paths["final"], exist_ok=True)
    return paths

def make_cycle_dirs(outdir, cycle_idx):
    cdir = os.path.join(outdir, "cycles", f"cycle_{cycle_idx:02d}")
    os.makedirs(cdir, exist_ok=True)
    return cdir

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

def plot_heat_combined(cdir, T_snap, r_grid, z_grid, N_r, N_z, cfg, tank_height, time_hr,
                       drain_r_idx, drain_angle, fill_r_idx, fill_angle):
    theta = np.linspace(0, 2*np.pi, 100)
    R_mesh, Theta_mesh = np.meshgrid(r_grid, theta)
    
    top_T = np.full(N_r, np.nan)
    for i in range(N_r):
        for j in range(N_z - 1, -1, -1):
            if not np.isnan(T_snap[i, j]):
                top_T[i] = T_snap[i, j]
                break
    T_mesh = np.tile(top_T, (len(theta), 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': None})
    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)

    cmap = plt.cm.inferno.copy(); cmap.set_bad('white')
    vmin, vmax = cfg["T_amb_low"], cfg["T_water_in"]

    c1 = ax1.pcolormesh(r_grid, z_grid, T_snap.T, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax1.axvspan(cfg["r_inner"], cfg["r_inner"] + cfg["t_wall"], color='dimgrey', alpha=0.8, lw=0)
    if cfg["t_ins"] > 0:
        ax1.axvspan(cfg["r_inner"] + cfg["t_wall"], cfg["r_inner"] + cfg["t_wall"] + cfg["t_ins"], color='lightgrey', alpha=0.8, lw=0, hatch='//')
    ax1.set_xlabel("Radius (m)"); ax1.set_ylabel("Height (m)")
    ax1.set_title(f"Lateral Temperature (t = {time_hr:.1f} h)"); ax1.set_xlim(0, cfg["r_inner"] + cfg["t_wall"] + cfg["t_ins"])
    ax1.axhline(cfg["coil_clearance_bottom"], color='cyan', linestyle='--', lw=1)
    ax1.axhline(cfg["coil_clearance_bottom"] + cfg["L_avail"], color='cyan', linestyle='--', lw=1)

    c2 = ax2.pcolormesh(Theta_mesh, R_mesh, T_mesh, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax2.set_title("Top-Down Surface Temp"); ax2.set_yticklabels([])

    if drain_r_idx > 0 or drain_angle > 0:
        d_theta = np.deg2rad(drain_angle)
        d_r = (drain_r_idx + 0.5) * (cfg["r_inner"] / N_r)
        ax2.plot(d_theta, d_r, 'rx', markersize=12, markeredgewidth=3, label='Drain')
    if fill_r_idx > 0 or fill_angle > 0:
        f_theta = np.deg2rad(fill_angle)
        f_r = (fill_r_idx + 0.5) * (cfg["r_inner"] / N_r)
        ax2.plot(f_theta, f_r, 'go', markersize=12, markerfacecolor='none', markeredgewidth=2.5, label='Fill')
    
    if (drain_r_idx > 0 or drain_angle > 0) or (fill_r_idx > 0 or fill_angle > 0):
        ax2.legend(loc='lower right', bbox_to_anchor=(1.1, -0.1), fontsize=9)

    fig.colorbar(c1, ax=[ax1, ax2], label='Temperature (°C)', pad=0.05)
    path = os.path.join(cdir, f"heatmap_hr_{time_hr:04.1f}.png")
    fig.savefig(path, dpi=100, bbox_inches='tight'); plt.close(fig)
    return path

def plot_mass_combined(cdir, M_snap, r_grid, z_grid, N_r, N_z, cfg, tank_height, time_hr,
                       drain_r_idx, drain_angle, fill_r_idx, fill_angle):
    theta = np.linspace(0, 2*np.pi, 100)
    R_mesh, Theta_mesh = np.meshgrid(r_grid, theta)
    
    r_edges = np.linspace(0, cfg["r_inner"], N_r + 1)
    col_vol = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2) * tank_height
    col_mass = np.nansum(M_snap, axis=1)
    col_density = col_mass / np.maximum(1e-9, col_vol)
    D_mesh = np.tile(col_density, (len(theta), 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': None})
    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)

    cmap_m = plt.cm.Blues.copy(); cmap_m.set_bad('white')
    c1 = ax1.pcolormesh(r_grid, z_grid, M_snap.T, cmap=cmap_m, shading='auto')
    ax1.axvspan(cfg["r_inner"], cfg["r_inner"] + cfg["t_wall"], color='dimgrey', alpha=0.8, lw=0)
    if cfg["t_ins"] > 0:
        ax1.axvspan(cfg["r_inner"] + cfg["t_wall"], cfg["r_inner"] + cfg["t_wall"] + cfg["t_ins"], color='lightgrey', alpha=0.8, lw=0, hatch='//')
    ax1.set_xlabel("Radius (m)"); ax1.set_ylabel("Height (m)")
    ax1.set_title(f"Lateral Mass (t = {time_hr:.1f} h)"); ax1.set_xlim(0, cfg["r_inner"] + cfg["t_wall"] + cfg["t_ins"])

    vmax_density = float(np.max(cfg["_mat"]["rho_grid"]))
    c2 = ax2.pcolormesh(Theta_mesh, R_mesh, D_mesh, cmap='Blues', shading='auto', vmin=0.0, vmax=vmax_density)
    ax2.set_title("Top-Down Column Density"); ax2.set_yticklabels([])

    if drain_r_idx > 0 or drain_angle > 0:
        d_theta = np.deg2rad(drain_angle)
        d_r = (drain_r_idx + 0.5) * (cfg["r_inner"] / N_r)
        ax2.plot(d_theta, d_r, 'rx', markersize=12, markeredgewidth=3, label='Drain')
    if fill_r_idx > 0 or fill_angle > 0:
        f_theta = np.deg2rad(fill_angle)
        f_r = (fill_r_idx + 0.5) * (cfg["r_inner"] / N_r)
        ax2.plot(f_theta, f_r, 'go', markersize=12, markerfacecolor='none', markeredgewidth=2.5, label='Fill')
    
    if (drain_r_idx > 0 or drain_angle > 0) or (fill_r_idx > 0 or fill_angle > 0):
        ax2.legend(loc='lower right', bbox_to_anchor=(1.1, -0.1), fontsize=9)

    fig.colorbar(c1, ax=ax1, label='Mass (kg)', pad=0.05)
    fig.colorbar(c2, ax=ax2, label='Density (kg/m³)', pad=0.05)
    path = os.path.join(cdir, f"massmap_hr_{time_hr:04.1f}.png")
    fig.savefig(path, dpi=100, bbox_inches='tight'); plt.close(fig)
    return path

def plot_phase_bar(cdir, cyc_sol_kg, cyc_slu_kg, cyc_liq_kg, cyc_sol_T, cyc_slu_T, cyc_liq_T, c):
    fig, ax = plt.subplots(figsize=(6, 4))
    phases = ['Solid', 'Slurry', 'Liquid']
    masses = [cyc_sol_kg, cyc_slu_kg, cyc_liq_kg]
    temps = [cyc_sol_T, cyc_slu_T, cyc_liq_T]
    colors = ['tab:gray', 'tab:orange', 'tab:blue']
    bars = ax.bar(phases, masses, color=colors)
    for idx, bar in enumerate(bars):
        if masses[idx] > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(masses)*0.02, 
                    f"{masses[idx]:.0f} kg\n{temps[idx]:.1f} °C", ha='center', fontsize=9)
    ax.set_ylabel("Mass (kg)"); ax.set_title(f"Cycle {c+1} — Phase Distribution")
    ax.grid(True, alpha=0.3, axis='y')
    path = os.path.join(cdir, "phasebar.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path

def write_cycle_summary(cdir, m, c):
    path = os.path.join(cdir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Cycle {c+1}\n")
        f.write(f"  Start mass    : {m['start_mass']:.1f} kg\n")
        f.write(f"  End-of-heat   : {m['remaining_mass']:.1f} kg\n")
        f.write(f"  Extracted     : {m['extracted_kg']:.1f} kg @ {m['extracted_temp_avg']:.1f} °C avg\n")
        f.write(f"  Filled        : {m['filled_kg']:.1f} kg\n")
        f.write(f"  Efficiency    : {m['efficiency_pct']:.1f} %\n")
        f.write(f"  Drain util    : {m['utilisation_pct']:.1f} %\n")
        f.write(f"  Fill util     : {m['fill_util_pct']:.1f} %\n")

def plot_flow_curve(fdir, time_log, drain_log, fill_log, is_heating_log, dt_sec, n_cycles):
    path = os.path.join(fdir, "flow_curve.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_log, drain_log, linewidth=2, color="tab:blue", label="Extracted")
    ax.plot(time_log, fill_log, linewidth=2, color="tab:orange", label="Filled")
    for step in range(len(time_log)):
        if not is_heating_log[step]:
            ax.axvspan(time_log[step], time_log[step] + dt_sec/3600.0, color='grey', alpha=0.2, lw=0)
    ax.legend(loc="upper left")
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Mass (kg)")
    ax.set_title(f"Mass Flow Curve ({n_cycles} Cycles)"); ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=130); plt.close(fig)
    return path

def write_final_summary(fdir, total_drained, total_filled, final_mass, first_drain, last_drain, total_hr):
    path = os.path.join(fdir, "final_summary.txt")
    first_t = f"{first_drain:.2f} h" if first_drain >= 0 else "Never"
    last_t = "Never" if last_drain < 0 else (f"{last_drain:.2f} h (Ongoing)" if last_drain >= total_hr - 0.01 else f"{last_drain:.2f} h")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Final Summary\n")
        f.write(f"  Total Drained : {total_drained:.1f} kg\n")
        f.write(f"  Total Filled  : {total_filled:.1f} kg\n")
        f.write(f"  Total Remaining: {final_mass:.1f} kg\n")
        f.write(f"  First drain   : {first_t}\n")
        f.write(f"  Last drain    : {last_t}\n")

# ------------------------------------------------------------
# HTML REPORT
# ------------------------------------------------------------
def _html_table(rows):
    return "<table>\n" + "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "\n</table>"

def _html_prop_arrays(name, T_arr, props):
    h = [f"<h3>{name} — properties at [{', '.join(f'{t:.1f}' for t in T_arr)}] degC</h3>", "<table><tr><th>Property</th>" + "".join(f"<th>{t:.1f} °C</th>" for t in T_arr) + "</tr>"]
    for label, vals, unit in props:
        h.append(f"<tr><td>{label}</td>" + "".join(f"<td>{v:g} {unit}</td>" for v in vals) + "</tr>")
    return "\n".join(h) + "\n</table>"

def _sample_arrays(T, rho, cp, mu, k, n=6):
    if len(T) <= n:
        idx = np.arange(len(T))
    else:
        idx = np.linspace(0, len(T)-1, n).astype(int)
    return T[idx], rho[idx], cp[idx], mu[idx], k[idx]

def _render_timeline(status_log, dt_sec, max_bins=400):
    if status_log is None: return "<p style='color:#888;'>Timeline not available.</p>"
    n = len(status_log)
    if n == 0: return "<p style='color:#888;'>No timeline data.</p>"
    if dt_sec is None or dt_sec <= 0: return "<p style='color:#888;'>Timeline dt unknown.</p>"
    total_dur = n * dt_sec / 3600.0

    if n > max_bins:
        bin_size = (n + max_bins - 1) // max_bins
        bins = []
        for b in range(0, n, bin_size):
            chunk = status_log[b:b + bin_size]
            if np.any(chunk == -1): st = -1
            elif np.any(chunk == 3): st = 3
            elif np.any(chunk == 1): st = 1
            elif np.any(chunk == 2): st = 2
            else: st = 0
            if bins and bins[-1][0] == st: bins[-1][1] += 1
            else: bins.append([st, 1])
        step_dur = bin_size * dt_sec / 3600.0
    else:
        bins = []
        for s in status_log:
            st = int(s)
            if bins and bins[-1][0] == st: bins[-1][1] += 1
            else: bins.append([st, 1])
        step_dur = dt_sec / 3600.0

    colors = {0: "#f39c12", 1: "#2ecc71", 2: "#3498db", 3: "#9b59b6", -1: "#95a5a6"}
    labels = {0: "Heating (Idle)", 1: "Draining", 2: "Filling", 3: "Drain + Fill", -1: "Cooling"}
    parts = []
    for st, cnt in bins:
        dur = cnt * step_dur
        if dur <= 0: continue
        parts.append(f'<div style="flex: {dur:.6f} 0 0; background:{colors.get(st, "#ccc")};" title="{labels.get(st, "?")} — {dur:.2f} h"></div>')
    
    legend = " &nbsp;&nbsp; ".join(
        f'<span style="display:inline-block; width:14px; height:14px; background:{colors[s]}; border-radius:2px; vertical-align:middle;"></span> {labels[s]}'
        for s in (3, 1, 2, 0, -1))
    return ('<div style="display:flex; width:100%; height:28px; border:1px solid #ccc; border-radius:4px; overflow:hidden; margin:8px 0;">'
            + "".join(parts) + "</div>"
            + f'<div style="font-size:12px; color:#555; margin-bottom:16px;">' + legend + f' &nbsp; (total {total_dur:.2f} h)</div>')

def write_report_html(outdir, cfg, R_ag, n_ag, results_2d=None):
    mat = cfg["_mat"]
    L_coil = 2.0*np.pi*R_ag*n_ag; A_coil = np.pi*cfg["d_out"]*L_coil
    rho_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["rho_tube_arr"]))
    M_metal = rho_tube_ref * np.pi*0.25*(cfg["d_out"]**2 - cfg["d_in"]**2)*L_coil
    rho_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    V_water = np.pi*0.25*cfg["d_in"]**2*L_coil

    T_start_str = f"{cfg['T_start']:.1f} °C" if 'T_start' in cfg else "n/a"
    M_eff_str = f"{cfg.get('M_fluid_effective', cfg['M_fluid']):.0f} kg"

    h = ["<!doctype html><html><head><meta charset='utf-8'><title>Tank Report</title>",
         "<style>body{font-family:sans-serif;max-width:1150px;margin:20px auto;color:#222}",
         "h1{border-bottom:2px solid #333;padding-bottom:6px} h2{margin-top:32px;color:#1a4d7a;border-bottom:1px solid #ddd;padding-bottom:4px}",
         "h3{margin-top:20px;color:#333;font-size:16px;} h4{margin:4px 0; font-size:14px; color:#555;}",
         "table{border-collapse:collapse;margin:8px 0 12px 0;font-size:13px} td,th{border:1px solid #ccc;padding:5px 8px;text-align:right}",
         "th{background:#eef4fa;text-align:center} td:first-child{text-align:left;font-weight:500}",
         ".banner{background:#1a4d7a; color:white; padding:12px; border-radius:4px; font-size:16px; margin-bottom:20px}",
         "img{max-width:100%;margin:4px 0;border:1px solid #ddd}",
         ".grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 16px; margin:12px 0;}",
         ".card{background:#f9f9f9; padding:10px; border-radius:6px; border:1px solid #eee;}",
         ".nodrain{color:#888; font-style:italic;}</style></head><body>",
         "<h1>Selected Coil — Detailed Report</h1>",
         f"<div class='banner'><b>Materials:</b> Coil Fluid={cfg['fluid_name']} | Tank Fluid={cfg['tank_name']} | Tube={cfg['coil_mat']} | Wall={cfg['wall_name']}</div>",
         f"<p><em>Run: {os.path.basename(outdir)}</em></p>"]

    h.append("<h2>1. Input Parameters</h2>")
    
    # Sample 6 points to display in the HTML table
    c_T, c_rho, c_cp, c_mu, c_k = _sample_arrays(cfg["T_water_arr"], cfg["rho_water_arr"], cfg["cp_water_arr"], cfg["mu_water_arr"], cfg["k_water_arr"])
    t_T, t_rho, t_cp, t_mu, t_k = _sample_arrays(mat["T_grid"], mat["rho_grid"], mat["cp_sens_grid"], mat["mu_grid"], mat["k_grid"])
    w_T, w_rho, w_cp, _, w_k = _sample_arrays(cfg["T_wall_arr"], cfg["rho_wall_arr"], cfg["cp_wall_arr"], np.zeros_like(cfg["T_wall_arr"]), cfg["k_wall_arr"])
    tb_T, tb_rho, tb_cp, _, tb_k = _sample_arrays(cfg["T_tube_arr"], cfg["rho_tube_arr"], cfg["cp_tube_arr"], np.zeros_like(cfg["T_tube_arr"]), cfg["k_tube_arr"])

    h.append(f"<h3>Coil Fluid: {cfg['fluid_name']}</h3>" + _html_prop_arrays("Coil", c_T, [("Density", c_rho, "kg/m³"), ("Cp", c_cp, "J/kgK"), ("Mu", c_mu, "Pa·s"), ("k", c_k, "W/mK")]))
    h.append(f"<h3>Tank Fluid: {cfg['tank_name']}</h3>" + _html_prop_arrays("Tank", t_T, [("Density", t_rho, "kg/m³"), ("Cp", t_cp, "J/kgK"), ("Mu", t_mu, "Pa·s"), ("k", t_k, "W/mK")]))
    h.append(f"<h3>Tank Wall: {cfg['wall_name']}</h3>" + _html_prop_arrays("Wall", w_T, [("Density", w_rho, "kg/m³"), ("Cp", w_cp, "J/kgK"), ("k", w_k, "W/mK")]))
    h.append(f"<h3>Coil Tube: {cfg['coil_mat']}</h3>" + _html_prop_arrays("Tube", tb_T, [("Density", tb_rho, "kg/m³"), ("Cp", tb_cp, "J/kgK"), ("k", tb_k, "W/mK")]))

    h.append("<h3>Geometry & Operating</h3>" + _html_table([
        ("Inner radius", f"{cfg['r_inner']:.4f} m"),
        ("Fluid mass (requested)", f"{cfg['M_fluid']:.0f} kg"),
        ("Starting tank temperature", T_start_str),
        ("Effective mass at T_start", M_eff_str),
        ("Environment (for heat loss)", f"{cfg['T_amb_mid']:.1f} °C"),
        ("Wall", f"{cfg['wall_name']}, t={cfg['t_wall']*1000:.1f} mm"),
        ("Insulation", f"t={cfg['t_ins']*1000:.0f} mm, U={cfg['h_air']:.1f} W/m²K"),
        ("Windows", f"Heating {cfg['t_window_hr']:.2f} h, Cooldown {cfg['t_cooldown_hr']:.2f} h"),
        ("Cycles", f"{cfg['N_cycles']}")
    ]))

    h.append("<h2>2. Selected Coil Geometry</h2>" + _html_table([
        ("Radius R", f"{R_ag:.4f} m"), ("Turns n", f"{n_ag}"), ("Developed length L", f"{L_coil:.2f} m"),
        ("Outer area A", f"{A_coil:.2f} m²"), ("Metal mass", f"{M_metal:.1f} kg"), ("Water volume", f"{V_water*1000:.1f} L")
    ]))

    if results_2d is not None:
        h.append("<h2>3. Simulation Results</h2>")
        h.append(f"<p><b>Total Simulation Time:</b> {results_2d['total_hr']:.1f} h</p>")

        h.append("<h2>Panel 1 — Cycle Details</h2>")
        for idx, m in enumerate(results_2d["cycle_metrics"]):
            c_num = idx + 1
            pb = f"cycles/cycle_{c_num:02d}/phasebar.png"
            
            h.append(f"<h3>Cycle {c_num}</h3>")
            
            temp_str = f"{m['extracted_temp_avg']:.1f} °C" if m["extracted_kg"] > 1e-6 else "<span class='nodrain'>—</span>"
            h.append("<table>")
            h.append(f"<tr><td>Fluid in tank (start of heating)</td><td>{m['start_mass']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Fluid in tank (end of heating)</td><td>{m['remaining_mass']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Fluid extracted via drain</td><td>{m['extracted_kg']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Average extracted temperature</td><td>{temp_str}</td></tr>")
            h.append(f"<tr><td>Drain efficiency</td><td>{m['efficiency_pct']:.1f} %</td></tr>")
            h.append(f"<tr><td>Drain utilisation</td><td>{m['utilisation_pct']:.1f} %</td></tr>")
            h.append(f"<tr><td>Fluid added via fill</td><td>{m['filled_kg']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Fill utilisation</td><td>{m['fill_util_pct']:.1f} %</td></tr>")
            h.append("</table>")

            h.append(f"<div class='card' style='max-width:400px;'><h4>Phase Distribution (End of Heating)</h4><img src='{pb}' alt='phasebar_{idx}'></div>")

            h.append("<h4>Hourly Snapshots</h4>")
            h.append("<div class='grid'>")
            for h_path, m_path, t_hr in zip(m["heatmaps"], m["massmaps"], m["snapshot_times"]):
                h.append(f"<div class='card'>")
                h.append(f"<h4>t = {t_hr:.1f} h</h4>")
                h.append(f"<img src='{h_path}' alt='heat_{idx}_{t_hr}'>")
                h.append(f"<img src='{m_path}' alt='mass_{idx}_{t_hr}'>")
                h.append("</div>")
            h.append("</div>")

        h.append("<h2>Panel 2 — Final Summary</h2>")
        h.append("<h3>Mass Flow Curve (Drain & Fill)</h3>")
        h.append(f"<img src='final/flow_curve.png' alt='curve'>")
        h.append("<h3>Timeline</h3>")
        h.append(_render_timeline(results_2d.get("status_log"), results_2d.get("dt_sec")))

        first_t = f"{results_2d['first_drain']:.2f} h" if results_2d['first_drain'] >= 0 else "Never"
        last_t = "Never" if results_2d['last_drain'] < 0 else (f"{results_2d['last_drain']:.2f} h (Ongoing)" if results_2d['last_drain'] >= results_2d['total_hr'] - 0.01 else f"{results_2d['last_drain']:.2f} h")

        h.append("<h3>Drain Summary</h3>")
        h.append("<table>")
        h.append(f"<tr><td>First drain activated</td><td>{first_t}</td></tr>")
        h.append(f"<tr><td>Drain stopped completely</td><td>{last_t}</td></tr>")
        h.append(f"<tr><td>Total Drained</td><td>{results_2d['total_extracted_kg']:.1f} kg</td></tr>")
        h.append(f"<tr><td>Total Filled</td><td>{results_2d['total_filled_kg']:.1f} kg</td></tr>")
        h.append(f"<tr><td>Total Remaining</td><td>{results_2d['final_mass']:.1f} kg</td></tr>")
        h.append("</table>")

    h.append("</body></html>")
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(h))