"""Post-processing: HTML, CSV, and graphs."""
import base64
import os
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from .physics import avg_T, wall_U_at_T
from .geometry import coil_properties, coil_index

CSV_KEYS = ["phase", "preset", "T_amb", "start_type", "R", "R_ratio", "n", "L_coil", "A_coil",
            "liquid_0", "slurry_0", "solid_0", "liquid", "slurry", "solid", "liquid_change",
            "liquid_T", "slurry_T", "solid_T", "peak_oil", "energy_kWh", "peak_power_kW"]
PNG_NAMES = ["cold_low.png", "cold_mid.png", "cold_hot.png", "hot_start.png", "selected_summary.png"]
CRLF = "\r\n"

def _viridis(n):
    cmap = matplotlib.colormaps["viridis"] if hasattr(matplotlib, "colormaps") else matplotlib.cm.get_cmap("viridis")
    return cmap(np.linspace(0.1, 0.9, n))

def write_csv(rows, path):
    if not rows: return
    lines = [",".join(CSV_KEYS)]
    for r in rows:
        d = dict(r); d["liquid_change"] = r["liquid"] - r.get("liquid_0", 0.0)
        lines.append(",".join(str(d.get(k, "")) for k in CSV_KEYS))
    with open(path, "w", encoding="utf-8", newline="") as f: f.write(CRLF.join(lines) + CRLF)

def plot_cold(rows, T_amb, out_path, t_window_hr):
    fig = Figure(figsize=(11, 9)); ax1, ax2 = fig.subplots(2, 1)
    radii = sorted(set(r["R"] for r in rows)); colors = _viridis(len(radii))
    for j, R in enumerate(radii):
        sub = sorted([r for r in rows if r["R"] == R], key=lambda x: x["n"]); L = [r["L_coil"] for r in sub]
        lbl = f"R={R:.2f}"
        ax1.plot(L, [r["liquid"] for r in sub], "-", color=colors[j], label=f"{lbl} L", linewidth=1.6)
        ax1.plot(L, [r["slurry"] for r in sub], "--", color=colors[j], label=f"{lbl} S", linewidth=1.2)
        ax1.plot(L, [r["solid"] for r in sub], ":", color=colors[j], label=f"{lbl} Sol", linewidth=1.2)
        ax2.plot(L, [r["liquid_T"] for r in sub], "-", color=colors[j], label=f"{lbl} liq T")
        ax2.plot(L, [r["slurry_T"] for r in sub], "--", color=colors[j], label=f"{lbl} slu T")
        ax2.plot(L, [r["solid_T"] for r in sub], ":", color=colors[j], label=f"{lbl} sol T")
    ax1.set_ylabel("Mass (kg)"); ax1.set_title(f"Cold start — ambient {T_amb} degC, {t_window_hr:.1f} h heating")
    ax1.legend(fontsize=7, ncol=3, loc="upper left"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Coil length (m)"); ax2.set_ylabel("Avg T (degC)")
    ax2.legend(fontsize=6, ncol=3, loc="best"); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)

def build_menu(radii, geom, cold_rows):
    menu = []
    for ri, R in enumerate(radii, 1):
        for g in sorted([g for g in geom if g["R"] == R], key=lambda x: x["n"]):
            menu.append({"idx": len(menu) + 1, "R_idx": ri, "R": g["R"], "n": g["n"], "L": g["L"], "R_ratio": g["R_ratio"]})
    lookup = {}
    for r in cold_rows:
        key = (round(r["R"], 6), r["n"])
        lookup.setdefault(key, {})[r["preset"]] = r["liquid"]
    for m in menu:
        vals = lookup.get((round(m["R"], 6), m["n"]), {})
        m["liq_low"] = vals.get("low", 0.0); m["liq_mid"] = vals.get("mid", 0.0); m["liq_hot"] = vals.get("hot", 0.0)
    return menu

def write_phase_a(outdir, cfg, cold_rows, geom, radii, presets):
    os.makedirs(outdir, exist_ok=True)
    for pname, Tamb in presets:
        rows = [r for r in cold_rows if r["preset"] == pname]
        plot_cold(rows, Tamb, os.path.join(outdir, f"cold_{pname}.png"), cfg["t_window_hr"])
    menu = build_menu(radii, geom, cold_rows)
    with open(os.path.join(outdir, "picker_menu.txt"), "w", encoding="utf-8", newline="") as f:
        lines = [f"{'idx':>4}  {'R(m)':>7}  {'n':>3}  {'L(m)':>7}  {'liq_low':>9}  {'liq_mid':>9}  {'liq_hot':>9}", "-" * 66]
        for m in menu:
            lines.append(f"{m['idx']:>4}  {m['R']:>7.3f}  {m['n']:>3}  {m['L']:>7.1f}  {m['liq_low']:>9.0f}  {m['liq_mid']:>9.0f}  {m['liq_hot']:>9.0f}")
        f.write(CRLF.join(lines) + CRLF)
    return menu

def _html_table(rows):
    out = ["<table>"]
    for k, v in rows: out.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    out.append("</table>"); return "\n".join(out)

def _html_prop_arrays(name, T_arr, props):
    h = [f"<h3>{name} — properties at [{', '.join(f'{t:.1f}' for t in T_arr)}] degC</h3>", "<table><tr><th>Property</th>" + "".join(f"<th>{t:.1f} °C</th>" for t in T_arr) + "</tr>"]
    for label, vals, unit in props:
        cells = "".join(f"<td>{v:g} {unit}</td>" for v in vals)
        h.append(f"<tr><td>{label}</td>{cells}</tr>")
    h.append("</table>"); return "\n".join(h)

def _png_data_uri(path):
    with open(path, "rb") as f: return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

def _T_band(r, cfg, mat, R_ag):
    return float(np.interp(np.array(r["H_final"])[coil_index(cfg, R_ag)], mat["H_grid"], mat["T_grid"]))

def write_report_html(outdir, cfg, R_ag, n_ag, cold_at_selected, cool_reports, hot_rows, presets, results_2d=None):
    mat = cfg["_mat"]; cp = coil_properties(cfg, R_ag, n_ag); U_ref = wall_U_at_T(cfg, cfg["T_water_in"])
    
    h = ["<!doctype html><html><head><meta charset='utf-8'><title>Tank Report</title>",
         "<style>body{font-family:sans-serif;max-width:1150px;margin:20px auto;color:#222}",
         "h1{border-bottom:2px solid #333;padding-bottom:6px} h2{margin-top:32px;color:#1a4d7a;border-bottom:1px solid #ddd;padding-bottom:4px}",
         "h3{margin-top:20px;color:#333;font-size:16px;} h4{margin:4px 0; font-size:14px; color:#555;}",
         "table{border-collapse:collapse;margin:8px 0 12px 0;font-size:13px} td,th{border:1px solid #ccc;padding:5px 8px;text-align:right}",
         "th{background:#eef4fa;text-align:center} td:first-child{text-align:left;font-weight:500}",
         ".banner{background:#1a4d7a; color:white; padding:12px; border-radius:4px; font-size:16px; margin-bottom:20px}",
         "img{max-width:100%;margin:12px 0;border:1px solid #ddd}",
         ".imgrow{display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; margin:8px 0;}",
         ".imgrow > div{flex:1 1 320px; min-width:280px;} .imgrow img{margin:4px 0;} .nodrain{color:#888; font-style:italic;}</style></head><body>",
         "<h1>Selected Coil — Detailed Report</h1>",
         f"<div class='banner'><b>Materials:</b> Coil Fluid={cfg['fluid_name']} | Tank Fluid={cfg['tank_name']} | Tube={cfg['coil_mat']} | Wall={cfg['wall_name']}</div>"]

    h.append("<h2>1. Input Parameters</h2>")
    h.append(f"<h3>Coil Fluid: {cfg['fluid_name']}</h3>" + _html_prop_arrays("Coil", cfg["T_water_arr"], [("Density", cfg["rho_water_arr"], "kg/m³"), ("Cp", cfg["cp_water_arr"], "J/kgK"), ("Mu", cfg["mu_water_arr"], "Pa·s"), ("k", cfg["k_water_arr"], "W/mK")]))
    h.append(f"<h3>Tank Fluid: {cfg['tank_name']}</h3>" + _html_prop_arrays("Solid", cfg["T_solid_arr"], [("Density", cfg["rho_solid_arr"], "kg/m³"), ("Cp", cfg["cp_solid_arr"], "J/kgK"), ("k", cfg["k_solid_arr"], "W/mK"), ("Mu", cfg["mu_solid_arr"], "Pa·s")]))
    h.append(_html_prop_arrays("Liquid", cfg["T_liquid_arr"], [("Density", cfg["rho_liquid_arr"], "kg/m³"), ("Cp", cfg["cp_liquid_arr"], "J/kgK"), ("k", cfg["k_liquid_arr"], "W/mK"), ("Mu", cfg["mu_liquid_arr"], "Pa·s")]))

    h.append("<h3>Geometry & Operating</h3>" + _html_table([
        ("Inner radius", f"{cfg['r_inner']:.4f} m"), ("Fluid mass (requested)", f"{cfg['M_fluid']:.0f} kg"),
        ("Starting tank temperature", f"{cfg.get('T_start', cfg['T_amb_mid']):.1f} °C"),
        ("Effective mass at T_start", f"{cfg.get('M_fluid_effective', cfg['M_fluid']):.0f} kg"),
        ("Environment (for heat loss)", f"{cfg['T_amb_mid']:.1f} °C"),
        ("Wall", f"{cfg['wall_name']}, t={cfg['t_wall']*1000:.1f} mm"),
        ("Insulation", f"t={cfg['t_ins']*1000:.0f} mm, U={cfg['h_air']:.1f} W/m²K"),
        ("Agitator", f"Type {cfg['ag_type']} @ {cfg['N_rpm']} RPM" if cfg['has_agitator'] else "None"),
        ("Windows", f"Heating {cfg['t_window_hr']:.2f} h, Cooldown {cfg['t_cooldown_hr']:.2f} h"),
        ("Cycles", f"{cfg['N_cycles']}"), ("Ambients", f"Cold: {cfg['T_amb_low']} / Amb: {cfg['T_amb_mid']} / Best: {cfg['T_amb_hot']} °C")
    ]))

    h.append("<h2>2. Selected Coil Geometry</h2>" + _html_table([
        ("Radius R", f"{R_ag:.4f} m"), ("Turns n", f"{n_ag}"), ("Developed length L", f"{cp['L_coil']:.2f} m"),
        ("Outer area A", f"{cp['A_coil']:.2f} m²"), ("Metal mass", f"{cp['M_metal']:.1f} kg"), ("Water volume", f"{cp['V_water']*1000:.1f} L")
    ]))

    if results_2d is not None:
        h.append("<h2>3. Simulation Results</h2>")
        h.append(f"<p><b>Total Simulation Time:</b> {results_2d['total_hr']:.1f} h</p>")
        h.append("<h3>Extraction Curve</h3>")
        h.append(f"<img src='{_png_data_uri(results_2d['curve_path'])}' alt='curve'>")
        
        first_t = f"{results_2d['first_drain']:.2f} h" if results_2d['first_drain'] >= 0 else "Never"
        last_t = "Never" if results_2d['last_drain'] < 0 else f"{results_2d['last_drain']:.2f} h"
        h.append("<table>")
        h.append(f"<tr><td>First drain activated</td><td>{first_t}</td></tr>")
        h.append(f"<tr><td>Drain stopped completely</td><td>{last_t}</td></tr>")
        h.append(f"<tr><td>Total Drained</td><td>{results_2d['total_extracted_kg']:.1f} kg</td></tr>")
        h.append(f"<tr><td>Total Remaining</td><td>{results_2d['final_mass']:.1f} kg</td></tr>")
        h.append("</table>")

        h.append("<h2>Panel 1 — Cycle Details</h2>")
        for idx, m in enumerate(results_2d["cycle_metrics"]):
            h.append(f"<h3>Cycle {m['cycle']}</h3>")
            h.append("<div class='imgrow'>"
                     f"<div><h4>Lateral Temperature</h4><img src='{_png_data_uri(results_2d['heatmap_paths'][idx])}'></div>"
                     f"<div><h4>Lateral Mass Distribution</h4><img src='{_png_data_uri(results_2d['massmap_paths'][idx])}'></div>"
                     "</div>")
            h.append("<div class='imgrow'>"
                     f"<div><h4>Top-Down Total Mass</h4><img src='{_png_data_uri(results_2d['topmass_paths'][idx])}'></div>"
                     f"<div><h4>Top-Down Surface Temp</h4><img src='{_png_data_uri(results_2d['toptemp_paths'][idx])}'></div>"
                     "</div>")
            h.append("<div class='imgrow'>"
                     f"<div><h4>Phase Distribution</h4><img src='{_png_data_uri(results_2d['phasebar_paths'][idx])}'></div>"
                     "</div>")
            h.append("<table>")
            h.append(f"<tr><td>Fluid in tank (start of heating)</td><td>{m['start_mass']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Fluid in tank (end of heating)</td><td>{m['remaining_mass']:.1f} kg</td></tr>")
            h.append(f"<tr><td>Fluid extracted via drain</td><td>{m['extracted_kg']:.1f} kg</td></tr>")
            h.append("</table>")

    h.append("</body></html>")
    path = os.path.join(outdir, "index.html")
    with open(path, "w", encoding="utf-8", newline="") as f: f.write("\n".join(h).replace("\n", CRLF))
    return path