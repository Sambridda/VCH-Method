"""Post-processing.

Phase A output (written first, before the user picks a coil):
    cold_low.png, cold_mid.png, cold_hot.png, picker_menu.txt
Final output (written once, after phase D):
    hot_start.png, selected_summary.png, results.csv, index.html
"""
import base64
import os

import numpy as np
import matplotlib
from matplotlib.figure import Figure

from .physics import H_to_T, avg_T, wall_U_at_T
from .geometry import coil_properties, coil_index

CSV_KEYS = ["phase", "preset", "T_amb", "start_type", "R", "R_ratio", "n", "L_coil", "A_coil",
            "liquid_0", "slurry_0", "solid_0",
            "liquid", "slurry", "solid", "liquid_change",
            "liquid_T", "slurry_T", "solid_T",
            "peak_oil", "energy_kWh", "peak_power_kW"]

PNG_NAMES = ["cold_low.png", "cold_mid.png", "cold_hot.png",
             "hot_start.png", "selected_summary.png"]

CRLF = "\r\n"


def _viridis(n):
    cmap = matplotlib.colormaps["viridis"] if hasattr(matplotlib, "colormaps") \
        else matplotlib.cm.get_cmap("viridis")
    return cmap(np.linspace(0.1, 0.9, n))


# ============================================================
# CSV
# ============================================================
def write_csv(rows, path):
    """Same 22 columns and Windows line endings as the reference results.csv."""
    if not rows:
        return
    lines = [",".join(CSV_KEYS)]
    for r in rows:
        d = dict(r)
        d["liquid_change"] = r["liquid"] - r.get("liquid_0", 0.0)
        lines.append(",".join(str(d.get(k, "")) for k in CSV_KEYS))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(CRLF.join(lines) + CRLF)


# ============================================================
# PLOTS (matplotlib Figure API: safe to call from a worker thread)
# ============================================================
def plot_cold(rows, T_amb, out_path, t_window_hr):
    fig = Figure(figsize=(11, 9))
    ax1, ax2 = fig.subplots(2, 1)
    radii = sorted(set(r["R"] for r in rows))
    colors = _viridis(len(radii))
    for j, R in enumerate(radii):
        sub = sorted([r for r in rows if r["R"] == R], key=lambda x: x["n"])
        L = [r["L_coil"] for r in sub]
        lbl = f"R={R:.2f}"
        ax1.plot(L, [r["liquid"] for r in sub], "-", color=colors[j], label=f"{lbl} L", linewidth=1.6)
        ax1.plot(L, [r["slurry"] for r in sub], "--", color=colors[j], label=f"{lbl} S", linewidth=1.2)
        ax1.plot(L, [r["solid"] for r in sub], ":", color=colors[j], label=f"{lbl} Sol", linewidth=1.2)
        ax2.plot(L, [r["liquid_T"] for r in sub], "-", color=colors[j], label=f"{lbl} liq T")
        ax2.plot(L, [r["slurry_T"] for r in sub], "--", color=colors[j], label=f"{lbl} slu T")
        ax2.plot(L, [r["solid_T"] for r in sub], ":", color=colors[j], label=f"{lbl} sol T")
    ax1.set_ylabel("Mass (kg)")
    ax1.set_title(f"Cold start — ambient {T_amb} degC, {t_window_hr:.1f} h heating — "
                  f"L=liquid, S=slurry, Sol=solid")
    ax1.legend(fontsize=7, ncol=3, loc="upper left"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Coil length (m)"); ax2.set_ylabel("Avg T (degC)")
    ax2.legend(fontsize=6, ncol=3, loc="best"); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)


def plot_hot(rows, out_path, t_window_hr):
    fig = Figure(figsize=(10, 9))
    ax1, ax2 = fig.subplots(2, 1)
    xs = list(range(len(rows))); xlabels = [f"{r['T_amb']}C" for r in rows]
    ax1.plot(xs, [r["liquid_0"] for r in rows], "o--", color="tab:blue", label="liquid initial", mfc="white", ms=9)
    ax1.plot(xs, [r["liquid"] for r in rows], "o-", color="tab:blue", label="liquid final", ms=9)
    ax1.plot(xs, [r["slurry_0"] for r in rows], "s--", color="tab:orange", label="slurry initial", mfc="white", ms=9)
    ax1.plot(xs, [r["slurry"] for r in rows], "s-", color="tab:orange", label="slurry final", ms=9)
    ax1.plot(xs, [r["solid_0"] for r in rows], "^--", color="tab:gray", label="solid initial", mfc="white", ms=9)
    ax1.plot(xs, [r["solid"] for r in rows], "^-", color="tab:gray", label="solid final", ms=9)
    ax1.set_xticks(xs); ax1.set_xticklabels(xlabels); ax1.set_ylabel("Mass (kg)")
    ax1.set_title(f"Hot run — dashed=initial (residual), solid=final (after {t_window_hr:.1f} h heating)")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(True, alpha=0.3)
    ax2.plot(xs, [r["liquid_T"] for r in rows], "o-", color="tab:blue", label="liquid T", ms=9)
    ax2.plot(xs, [r["slurry_T"] for r in rows], "s-", color="tab:orange", label="slurry T", ms=9)
    ax2.plot(xs, [r["solid_T"] for r in rows], "^-", color="tab:gray", label="solid T", ms=9)
    ax2.set_xticks(xs); ax2.set_xticklabels(xlabels)
    ax2.set_xlabel("Ambient temperature"); ax2.set_ylabel("Avg T (degC)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130)


def plot_selected_summary(cold_at_selected, cool_reports, hot_rows, presets,
                          out_path, R_ag, n_ag, t_window_hr, t_cooldown_hr):
    fig = Figure(figsize=(16, 6))
    axes = fig.subplots(1, 3)
    labels = [f"{p}\n{T} C" for p, T in presets]
    x = np.arange(3)
    phases = [("liquid", "tab:blue", "liquid"),
              ("slurry", "tab:orange", "slurry"),
              ("solid", "tab:gray", "solid")]

    def draw(ax, rows, title):
        bottoms = np.zeros(3)
        for key, color, label in phases:
            vals = np.array([r[key] for r in rows])
            ax.bar(x, vals, 0.6, bottom=bottoms, color=color, label=label, alpha=0.85)
            bottoms += vals
        for i, r in enumerate(rows):
            tot = r["liquid"] + r["slurry"] + r["solid"]
            Ta = ((r["liquid"] * r["liquid_T"] + r["slurry"] * r["slurry_T"]
                   + r["solid"] * r["solid_T"]) / tot if tot > 0 else 0)
            ax.text(i, bottoms[i] * 1.02 if bottoms[i] > 0 else 0.02 * max(1, bottoms.max()),
                    f"{Ta:.0f}C", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Mass (kg)")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    draw(axes[0], [cold_at_selected[p] for p, _ in presets],
         f"Cold start — end of {t_window_hr:.1f} h heating")
    draw(axes[1], [cool_reports[p] for p, _ in presets],
         f"Residual — after {t_cooldown_hr:.1f} h no power")
    draw(axes[2], hot_rows,
         f"Hot run — end of {t_window_hr:.1f} h heating from residual")
    L_coil = 2.0 * np.pi * R_ag * n_ag
    fig.suptitle(f"Selected coil: R={R_ag:.3f} m, n={n_ag} turns, L={L_coil:.1f} m",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=130)


# ============================================================
# PHASE A OUTPUT (written first)
# ============================================================
def build_menu(radii, geom, cold_rows):
    menu = []
    for ri, R in enumerate(radii, 1):
        for g in sorted([g for g in geom if g["R"] == R], key=lambda x: x["n"]):
            menu.append({"idx": len(menu) + 1, "R_idx": ri, "R": g["R"], "n": g["n"],
                         "L": g["L"], "R_ratio": g["R_ratio"]})
    lookup = {}
    for r in cold_rows:
        key = (round(r["R"], 6), r["n"])
        lookup.setdefault(key, {})[r["preset"]] = r["liquid"]
    for m in menu:
        vals = lookup.get((round(m["R"], 6), m["n"]), {})
        m["liq_low"] = vals.get("low", 0.0)
        m["liq_mid"] = vals.get("mid", 0.0)
        m["liq_hot"] = vals.get("hot", 0.0)
    return menu


def write_phase_a(outdir, cfg, cold_rows, geom, radii, presets):
    os.makedirs(outdir, exist_ok=True)
    for pname, Tamb in presets:
        rows = [r for r in cold_rows if r["preset"] == pname]
        plot_cold(rows, Tamb, os.path.join(outdir, f"cold_{pname}.png"), cfg["t_window_hr"])

    menu = build_menu(radii, geom, cold_rows)
    with open(os.path.join(outdir, "picker_menu.txt"), "w", encoding="utf-8", newline="") as f:
        lines = [f"{'idx':>4}  {'R(m)':>7}  {'n':>3}  {'L(m)':>7}  "
                 f"{'liq_low':>9}  {'liq_mid':>9}  {'liq_hot':>9}",
                 "-" * 66]
        for m in menu:
            lines.append(f"{m['idx']:>4}  {m['R']:>7.3f}  {m['n']:>3}  {m['L']:>7.1f}  "
                         f"{m['liq_low']:>9.0f}  {m['liq_mid']:>9.0f}  {m['liq_hot']:>9.0f}")
        f.write(CRLF.join(lines) + CRLF)
    return menu


# ============================================================
# FINAL REPORT
# ============================================================
def _html_table(rows):
    out = ["<table>"]
    for k, v in rows:
        out.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def _html_prop_arrays(name, T_arr, props):
    h = [f"<h3>{name} — properties at [{', '.join(f'{t:.1f}' for t in T_arr)}] degC</h3>",
         "<table><tr><th>Property</th>"
         + "".join(f"<th>{t:.1f} °C</th>" for t in T_arr) + "</tr>"]
    for label, vals, unit in props:
        cells = "".join(f"<td>{v:g} {unit}</td>" for v in vals)
        h.append(f"<tr><td>{label}</td>{cells}</tr>")
    h.append("</table>")
    return "\n".join(h)


def _png_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _T_band(r, cfg, mat, R_ag):
    return float(np.interp(np.array(r["H_final"])[coil_index(cfg, R_ag)], mat["H_grid"], mat["T_grid"]))


def write_report_html(outdir, cfg, R_ag, n_ag, cold_at_selected,
                      cool_reports, hot_rows, presets, pngs):
    mat = cfg["_mat"]
    cp = coil_properties(cfg, R_ag, n_ag)
    U_ref = wall_U_at_T(cfg, cfg["T_water_in"])

    h = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<title>P6 Selected Coil Report</title>",
         "<style>body{font-family:sans-serif;max-width:1150px;margin:20px auto;color:#222}",
         "h1{border-bottom:2px solid #333;padding-bottom:6px}",
         "h2{margin-top:32px;color:#1a4d7a;border-bottom:1px solid #ddd;padding-bottom:4px}",
         "h3{margin-top:18px;color:#333;font-size:15px;margin-bottom:4px}",
         "table{border-collapse:collapse;margin:8px 0 12px 0;font-size:13px}",
         "td,th{border:1px solid #ccc;padding:5px 8px;text-align:right}",
         "th{background:#eef4fa;text-align:center}",
         "td:first-child{text-align:left;font-weight:500}",
         ".net-loss{color:#c00;font-weight:bold}",
         "img{max-width:100%;margin:12px 0;border:1px solid #ddd}",
         "</style></head><body>",
         "<h1>Selected Coil — Detailed Report</h1>",
         f"<p><em>Run: {os.path.basename(os.path.normpath(outdir))}</em></p>"]

    h.append("<h2>1. Input Parameters</h2>")
    h.append(f"<h3>Coil Fluid: {cfg['fluid_name']} (band {cfg['T_water_arr'][0]:.1f}–{cfg['T_water_arr'][-1]:.1f} °C)</h3>")
    h.append(_html_prop_arrays(cfg["fluid_name"], cfg["T_water_arr"], [
        ("Density", cfg["rho_water_arr"], "kg/m³"),
        ("Specific heat", cfg["cp_water_arr"], "J/(kg·K)"),
        ("Viscosity", cfg["mu_water_arr"], "Pa·s"),
        ("Thermal cond.", cfg["k_water_arr"], "W/(m·K)"),
    ]))
    h.append(f"<h3>Tank Fluid: {cfg['tank_name']} (melt {cfg['T_ms']:.1f}–{cfg['T_me']:.1f} °C, Hfus {cfg['Hfus']:.0f} J/kg)</h3>")
    h.append(_html_prop_arrays("Solid", cfg["T_solid_arr"], [
        ("Density", cfg["rho_solid_arr"], "kg/m³"),
        ("Specific heat", cfg["cp_solid_arr"], "J/(kg·K)"),
        ("Thermal cond.", cfg["k_solid_arr"], "W/(m·K)"),
        ("Viscosity", cfg["mu_solid_arr"], "Pa·s"),
    ]))
    h.append(_html_prop_arrays("Liquid", cfg["T_liquid_arr"], [
        ("Density", cfg["rho_liquid_arr"], "kg/m³"),
        ("Specific heat", cfg["cp_liquid_arr"], "J/(kg·K)"),
        ("Thermal cond.", cfg["k_liquid_arr"], "W/(m·K)"),
        ("Viscosity", cfg["mu_liquid_arr"], "Pa·s"),
    ]))
    h.append(f"<h3>Tank Wall: {cfg['wall_name']} (t = {cfg['t_wall']*1000:.1f} mm)</h3>")
    h.append(_html_prop_arrays("Wall", cfg["T_wall_arr"], [
        ("Density", cfg["rho_wall_arr"], "kg/m³"),
        ("Specific heat", cfg["cp_wall_arr"], "J/(kg·K)"),
        ("Thermal cond.", cfg["k_wall_arr"], "W/(m·K)"),
    ]))
    h.append(f"<h3>Coil Tube: {cfg['coil_mat']} (d_out {cfg['d_out']*1000:.1f} mm, d_in {cfg['d_in']*1000:.1f} mm, Rf_out {cfg['R_f_out']:.5f} m²·K/W)</h3>")
    h.append(_html_prop_arrays("Tube", cfg["T_tube_arr"], [
        ("Density", cfg["rho_tube_arr"], "kg/m³"),
        ("Specific heat", cfg["cp_tube_arr"], "J/(kg·K)"),
        ("Thermal cond.", cfg["k_tube_arr"], "W/(m·K)"),
    ]))

    h.append("<h3>Insulation / Environment</h3>")
    h.append(_html_table([
        ("Insulation thickness", f"{cfg['t_ins']:.4f} m ({cfg['t_ins']*1000:.0f} mm)"),
        ("Insulation conductivity", f"{cfg['k_ins']:.3f} W/(m·K)"),
        ("External convection h_air", f"{cfg['h_air']:.1f} W/(m²·K)"),
        ("Overall U (at T_water_in)", f"{U_ref:.3f} W/(m²·K)"),
    ]))

    h.append("<h3>Tank Geometry / Mass / Operating</h3>")
    rows = [
        ("Inner radius", f"{cfg['r_inner']:.4f} m"),
        ("Coil axial length", f"{cfg['L_avail']:.4f} m"),
        ("Tank fluid mass", f"{cfg['M_fluid']:.0f} kg"),
        ("Coil inlet temperature", f"{cfg['T_water_in']:.1f} °C"),
        ("Coil volumetric flow", f"{cfg['flow_m3h']:.3f} m³/h"),
        ("Heating window", f"{cfg['t_window_hr']:.2f} h"),
        ("Cooldown window", f"{cfg['t_cooldown_hr']:.2f} h"),
        ("Timestep dt", f"{cfg['dt_sec']} s"),
        ("Low / Mid / Hot ambient", f"{cfg['T_amb_low']:.1f} / {cfg['T_amb_mid']:.1f} / {cfg['T_amb_hot']:.1f} °C"),
        ("Radial nodes N", f"{cfg['N']}"),
        ("Turn samples per radius", f"{cfg['n_samples']}"),
    ]
    if cfg["has_agitator"]:
        rows.extend([
            ("Agitator Type", f"Type {cfg['ag_type']} ({cfg['N_rpm']} RPM, D={cfg['D_imp']:.2f} m)"),
            ("Agitator Visc. Corr.", f"{cfg['visc_corr']}"),
        ])
    else:
        rows.append(("Agitator", "None"))
    h.append(_html_table(rows))

    h.append("<h2>2. Selected Coil Geometry</h2>")
    h.append(_html_table([
        ("Radius R", f"{R_ag:.4f} m ({R_ag/cfg['r_inner']:.3f} × r_tank)"),
        ("Turns n", f"{n_ag}"),
        ("Helix pitch", f"{cfg['d_out']:.4f} m (= tube OD)"),
        ("Developed length L", f"{cp['L_coil']:.2f} m"),
        ("Outer surface area A", f"{cp['A_coil']:.2f} m²"),
        ("Axial height used", f"{cp['axial_height']:.3f} m (of {cfg['L_avail']:.3f} m)"),
        ("Tube metal mass", f"{cp['M_metal']:.1f} kg ({cfg['coil_mat']})"),
        ("Coil water volume", f"{cp['V_water']*1000:.1f} L  ({cp['M_water']:.1f} kg)"),
    ]))

    h.append(f"<h2>3. Cold Start — {cfg['t_window_hr']:.1f} h heating from ambient</h2><table>")
    h.append("<tr><th>Ambient</th><th>T_amb (°C)</th><th>liquid (kg)</th><th>slurry (kg)</th>"
             "<th>solid (kg)</th><th>T_avg (°C)</th><th>peak kW</th><th>energy kWh</th></tr>")
    for pname, Tamb in presets:
        r = cold_at_selected[pname]
        h.append(f"<tr><td>{pname}</td><td>{Tamb:.1f}</td>"
                 f"<td>{r['liquid']:.0f}</td><td>{r['slurry']:.0f}</td><td>{r['solid']:.0f}</td>"
                 f"<td>{avg_T(r):.1f}</td><td>{r['peak_power_kW']:.1f}</td>"
                 f"<td>{r['energy_kWh']:.1f}</td></tr>")
    h.append("</table>")

    h.append(f"<h2>4. Cooldown — {cfg['t_cooldown_hr']:.1f} h no power (residual state)</h2><table>")
    h.append("<tr><th>Ambient</th><th>T_band (°C)</th><th>T_steel (°C)</th>"
             "<th>liquid (kg)</th><th>slurry (kg)</th><th>solid (kg)</th><th>T_avg (°C)</th></tr>")
    for pname, _ in presets:
        r = cool_reports[pname]
        h.append(f"<tr><td>{pname}</td><td>{_T_band(r, cfg, mat, R_ag):.1f}</td><td>{r['T_steel']:.1f}</td>"
                 f"<td>{r['liquid']:.0f}</td><td>{r['slurry']:.0f}</td><td>{r['solid']:.0f}</td>"
                 f"<td>{avg_T(r):.1f}</td></tr>")
    h.append("</table>")

    h.append(f"<h2>5. Hot Run — {cfg['t_window_hr']:.1f} h heating from residual state</h2><table>")
    h.append("<tr><th>Ambient</th><th>liq init (kg)</th><th>liq final (kg)</th>"
             "<th>change (kg)</th><th>T_avg (°C)</th><th>peak kW</th><th>energy kWh</th></tr>")
    for r in hot_rows:
        chg = r["liquid"] - r["liquid_0"]
        cls = ' class="net-loss"' if chg < -1 else ""
        h.append(f"<tr><td>{r['preset']}</td><td>{r['liquid_0']:.0f}</td>"
                 f"<td>{r['liquid']:.0f}</td><td{cls}>{chg:+.0f}</td>"
                 f"<td>{avg_T(r):.1f}</td><td>{r['peak_power_kW']:.1f}</td>"
                 f"<td>{r['energy_kWh']:.1f}</td></tr>")
    h.append("</table>")

    h.append("<h2>6. Graphs</h2>")
    for p in pngs:
        h.append(f"<h3>{p}</h3><img src='{_png_data_uri(os.path.join(outdir, p))}' alt='{p}'>")
    h.append("</body></html>")

    path = os.path.join(outdir, "index.html")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(h).replace("\n", CRLF))
    return path


def final_report_text(cfg, R_ag, n_ag, cold_at_selected, cool_reports, hot_rows, presets):
    mat = cfg["_mat"]
    cp = coil_properties(cfg, R_ag, n_ag)
    U_ref = wall_U_at_T(cfg, cfg["T_water_in"])
    L = []
    L.append("FINAL REPORT — SELECTED COIL DIMENSION")
    L.append("=" * 70)
    L.append("\n COIL PROPERTIES")
    L.append(f"   Radius R            : {R_ag:.4f} m  ({R_ag/cfg['r_inner']:.3f} x r_tank)")
    L.append(f"   Turns n             : {n_ag}")
    L.append(f"   Helix pitch         : {cfg['d_out']:.4f} m (= tube OD)")
    L.append(f"   Developed length L  : {cp['L_coil']:.2f} m")
    L.append(f"   Outer area A        : {cp['A_coil']:.2f} m2")
    L.append(f"   Axial height used   : {cp['axial_height']:.3f} m  (of {cfg['L_avail']:.3f} m)")
    L.append(f"   Tube d_out / d_in   : {cfg['d_out']:.5f} / {cfg['d_in']:.5f} m")
    L.append(f"   Metal mass          : {cp['M_metal']:.1f} kg ({cfg['coil_mat']})")
    L.append(f"   Water volume        : {cp['V_water']*1000:.1f} L  ({cp['M_water']:.1f} kg)")
    L.append("\n TANK & OPERATING")
    L.append(f"   Inner radius        : {cfg['r_inner']:.3f} m")
    L.append(f"   Fluid mass          : {cfg['M_fluid']:.0f} kg ({cfg['tank_name']})")
    L.append(f"   Wall                : {cfg['wall_name']}, t={cfg['t_wall']*1000:.1f} mm")
    L.append(f"   Insulation          : t={cfg['t_ins']*1000:.0f} mm, U at T_in = {U_ref:.3f} W/(m2*K)")
    L.append(f"   Coil inlet T        : {cfg['T_water_in']:.1f} degC ({cfg['fluid_name']})")
    L.append(f"   Coil flow           : {cfg['flow_m3h']:.2f} m3/h")
    if cfg["has_agitator"]:
        L.append(f"   Agitator            : Type {cfg['ag_type']}  N={cfg['N_rpm']} RPM  D={cfg['D_imp']:.2f} m")
    else:
        L.append("   Agitator            : None")
    L.append(f"   Heating window      : {cfg['t_window_hr']:.2f} h  (dt={cfg['dt_sec']} s)")
    L.append(f"   Cooldown window     : {cfg['t_cooldown_hr']:.2f} h")

    L.append(f"\n COLD START — {cfg['t_window_hr']:.1f} h heating from ambient")
    L.append(f"   {'Amb':>4} {'T_amb':>6} {'Liquid':>8} {'Slurry':>8} {'Solid':>8} {'T_avg':>7} {'Pk kW':>7} {'kWh':>8}")
    for pname, Tamb in presets:
        r = cold_at_selected[pname]
        L.append(f"   {pname:>4} {Tamb:>6.1f} {r['liquid']:>8.0f} {r['slurry']:>8.0f} "
                 f"{r['solid']:>8.0f} {avg_T(r):>7.1f} {r['peak_power_kW']:>7.1f} {r['energy_kWh']:>8.1f}")

    L.append(f"\n COOLDOWN — {cfg['t_cooldown_hr']:.1f} h no power (residual state)")
    L.append(f"   {'Amb':>4} {'T_band':>7} {'T_steel':>8} {'Liquid':>8} {'Slurry':>8} {'Solid':>8} {'T_avg':>7}")
    for pname, _ in presets:
        r = cool_reports[pname]
        L.append(f"   {pname:>4} {_T_band(r, cfg, mat, R_ag):>7.1f} {r['T_steel']:>8.1f} {r['liquid']:>8.0f} "
                 f"{r['slurry']:>8.0f} {r['solid']:>8.0f} {avg_T(r):>7.1f}")

    L.append(f"\n HOT RUN — {cfg['t_window_hr']:.1f} h heating from residual state")
    L.append(f"   {'Amb':>4} {'liq init':>9} {'liq final':>10} {'change':>8} {'T_avg':>7} {'Pk kW':>7} {'kWh':>8}")
    for r in hot_rows:
        chg = r["liquid"] - r["liquid_0"]
        flag = "  [NET LOSS]" if chg < -1 else ""
        L.append(f"   {r['preset']:>4} {r['liquid_0']:>9.0f} {r['liquid']:>10.0f} "
                 f"{chg:>+8.0f} {avg_T(r):>7.1f} {r['peak_power_kW']:>7.1f} "
                 f"{r['energy_kWh']:>8.1f}{flag}")
    return "\n".join(L)


def write_final(outdir, cfg, R_ag, n_ag, cold_rows, cold_at_selected,
                cool_reports, hot_rows, presets):
    plot_hot(hot_rows, os.path.join(outdir, "hot_start.png"), cfg["t_window_hr"])
    plot_selected_summary(cold_at_selected, cool_reports, hot_rows, presets,
                          os.path.join(outdir, "selected_summary.png"),
                          R_ag, n_ag, cfg["t_window_hr"], cfg["t_cooldown_hr"])
    csv_path = os.path.join(outdir, "results.csv")
    write_csv(list(cold_rows) + list(hot_rows), csv_path)
    html_path = write_report_html(outdir, cfg, R_ag, n_ag, cold_at_selected,
                                  cool_reports, hot_rows, presets, PNG_NAMES)
    return dict(results_csv=csv_path, index_html=html_path,
                text=final_report_text(cfg, R_ag, n_ag, cold_at_selected,
                                       cool_reports, hot_rows, presets))