# Main.py — Orchestration only
import os
import time
import concurrent.futures
import multiprocessing
from datetime import datetime
import numpy as np
import ui
import outputs
import materials
from physics import engine_1d, engine_2d

def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "mat"

def main():
    cfg = ui.gather_initial_inputs()
    
    # Load materials from cache
    mat_tank = materials.load(cfg["tank_name"])
    mat_coil = materials.load(cfg["fluid_name"])
    mat_tube = materials.load(cfg["coil_mat"])
    mat_wall = materials.load(cfg["wall_name"])
    
    # For backwards compatibility, the engines expect `mat` to be the tank fluid
    mat = mat_tank 
    cfg["_mat"] = mat

    # Flatten the other materials into cfg for the engines to access
    cfg["T_water_arr"] = mat_coil["T_grid"]
    cfg["rho_water_arr"] = mat_coil["rho_grid"]
    cfg["cp_water_arr"] = mat_coil["cp_sens_grid"]
    cfg["mu_water_arr"] = mat_coil["mu_grid"]
    cfg["k_water_arr"] = mat_coil["k_grid"]
    
    cfg["T_tube_arr"] = mat_tube["T_grid"]
    cfg["rho_tube_arr"] = mat_tube["rho_grid"]
    cfg["cp_tube_arr"] = mat_tube["cp_sens_grid"]
    cfg["k_tube_arr"] = mat_tube["k_grid"]
    
    cfg["T_wall_arr"] = mat_wall["T_grid"]
    cfg["rho_wall_arr"] = mat_wall["rho_grid"]
    cfg["cp_wall_arr"] = mat_wall["cp_sens_grid"]
    cfg["k_wall_arr"] = mat_wall["k_grid"]
    
    cfg["T_ms"] = mat_tank["T_ms"]
    cfg["T_me"] = mat_tank["T_me"]
    
    # Remove agitator flags for the engines
    cfg["has_agitator"] = 0
    cfg["ag_type"] = 0
    cfg["N_rpm"] = 0.0
    cfg["D_imp"] = 0.0
    cfg["visc_corr"] = 1.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("out", f"run_{ts}_{sanitize_filename(cfg['fluid_name'])}_{sanitize_filename(cfg['tank_name'])}")
    os.makedirs(outdir, exist_ok=True)
    print(f"\nOutput: {outdir}")

    # Phase A
    geom, n_max, radii, turns = engine_1d.build_geometry(cfg)
    presets = [("expected_cold", cfg["T_amb_low"]), ("ambient", cfg["T_amb_mid"]), ("best_case", cfg["T_amb_hot"])]

    print(f"\n--- Phase A: cold-start sweep ({len(radii)} radii x {len(turns)} turns) ---")
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
            ui.progress(done_A, total_A, t_A, prefix="Phase A ")

    for pname, Tamb in presets:
        rows = [r for r in cold_rows_all if r["preset"] == pname]
        outputs.plot_cold(rows, Tamb, os.path.join(outdir, f"cold_{pname}.png"), cfg["t_window_hr"], pname.replace("_", " ").title())

    # Phase B
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
    R_ag, n_ag = ui.pick_coil(menu)

    # Phase C
    cfg = ui.gather_2d_inputs(cfg, mat_tank)

    # Phase D
    print("\n" + "="*70)
    print(" PHASE D: 2D SIMULATION")
    print("="*70)

    results_2d = engine_2d.run_2d_simulation(
        cfg, mat, R_ag, n_ag, None, 
        cfg["drain_diameter"], (cfg["drain_r_idx"], cfg["drain_z_idx"]), 
        outdir, on_cycle=ui.print_cycle
    )

    ui.print_final(results_2d, outdir)

if __name__ == "__main__":
    main()