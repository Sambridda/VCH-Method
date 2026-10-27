"""1D Sweep and Cooldown engines."""
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import BrokenExecutor
import numpy as np
from .core import _jit, calc_Q, pack_result
from ..geometry import prepare_mesh, injection_weights, steel_mass, tank_outer_area

class Cancelled(Exception): pass

@_jit
def transient_kernel(H, T_steel, N, dr, dt_inv_m, A_face, inj_w, T_ms, T_me, T_grid, H_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr, m_dot, T_water_in, d_in, d_out, R_f_out, A_coil, d_coil_actual, T_tube_arr, k_tube_arr, M_steel, T_wall_arr, k_wall_arr, cp_wall_arr, A_tank_outer, t_wall, t_ins, k_ins, h_air, T_amb, dt, n_steps, has_agitator, ag_C, ag_a, N_rpm, D_imp, visc_corr, beta, dT_natural):
    peak_oil = -1e9; Q_total = 0.0; peak_Q = 0.0
    T = np.empty(N); k_arr = np.empty(N)
    for i in range(N):
        Ti = np.interp(H[i], H_grid, T_grid)
        if Ti > peak_oil: peak_oil = Ti
    for step in range(n_steps):
        for i in range(N):
            T[i] = np.interp(H[i], H_grid, T_grid)
            if T[i] > peak_oil: peak_oil = T[i]
        for i in range(N):
            k_arr[i] = np.interp(T[i], T_grid, k_grid)
        step_Q = 0.0
        for i in range(N):
            if inj_w[i] > 0.0:
                Q_local = calc_Q(m_dot, T[i], T_water_in, d_in, d_out, R_f_out, A_coil, d_coil_actual, T_water_arr, rho_water_arr, cp_water_arr, mu_water_arr, k_water_arr, T_tube_arr, k_tube_arr, T_grid, rho_grid, cp_sens_grid, k_grid, mu_grid, T_ms, has_agitator, ag_C, ag_a, N_rpm, D_imp, visc_corr, beta, dT_natural)
                H[i] += Q_local * inj_w[i] * dt_inv_m[i]
                Q_total += Q_local * inj_w[i] * dt
                step_Q += Q_local * inj_w[i]
        if step_Q > peak_Q: peak_Q = step_Q
        for i in range(N - 1):
            kavg = 0.5 * (k_arr[i] + k_arr[i + 1])
            Qc = kavg * A_face[i] * (T[i] - T[i + 1]) / dr
            H[i] -= Qc * dt_inv_m[i]; H[i + 1] += Qc * dt_inv_m[i + 1]
        Q_os = k_arr[N - 1] * A_tank_outer * (T[N - 1] - T_steel) / (dr * 0.5)
        H[N - 1] -= Q_os * dt_inv_m[N - 1]
        k_wall_now = np.interp(T_steel, T_wall_arr, k_wall_arr); cp_wall_now = np.interp(T_steel, T_wall_arr, cp_wall_arr)
        R_conv = 1.0 / h_air; R_wall_half = (t_wall / 2.0) / k_wall_now
        R_ins = (t_ins / k_ins) if t_ins > 0.0 else 0.0
        U_air = 1.0 / (R_wall_half + R_ins + R_conv)
        Q_sa = U_air * A_tank_outer * (T_steel - T_amb)
        T_steel += (Q_os - Q_sa) * dt / (M_steel * cp_wall_now)
    for i in range(N):
        Ti = np.interp(H[i], H_grid, T_grid)
        if Ti > peak_oil: peak_oil = Ti
    return H, T_steel, peak_oil, Q_total, peak_Q

@_jit
def cooldown_kernel(H, T_steel, N, dr, dt_inv_m_eff, A_face, T_grid, H_grid, k_grid, M_steel, T_wall_arr, k_wall_arr, cp_wall_arr, A_tank_outer, t_wall, t_ins, k_ins, h_air, T_amb, dt, n_steps):
    T = np.empty(N); k_arr = np.empty(N)
    for step in range(n_steps):
        for i in range(N): T[i] = np.interp(H[i], H_grid, T_grid)
        for i in range(N): k_arr[i] = np.interp(T[i], T_grid, k_grid)
        for i in range(N - 1):
            kavg = 0.5 * (k_arr[i] + k_arr[i + 1])
            Qc = kavg * A_face[i] * (T[i] - T[i + 1]) / dr
            H[i] -= Qc * dt_inv_m_eff[i]; H[i + 1] += Qc * dt_inv_m_eff[i + 1]
        Q_os = k_arr[N - 1] * A_tank_outer * (T[N - 1] - T_steel) / (dr * 0.5)
        H[N - 1] -= Q_os * dt_inv_m_eff[N - 1]
        k_wall_now = np.interp(T_steel, T_wall_arr, k_wall_arr); cp_wall_now = np.interp(T_steel, T_wall_arr, cp_wall_arr)
        R_conv = 1.0 / h_air; R_wall_half = (t_wall / 2.0) / k_wall_now
        R_ins = (t_ins / k_ins) if t_ins > 0.0 else 0.0
        U_air = 1.0 / (R_wall_half + R_ins + R_conv)
        Q_sa = U_air * A_tank_outer * (T_steel - T_amb)
        T_steel += (Q_os - Q_sa) * dt / (M_steel * cp_wall_now)
    return H, T_steel

def run_heat(cfg, mat, R_coil, n_turns, T_amb, T_initial, t_hr):
    N = cfg["N"]; T_ref_mesh = T_amb if T_initial is None else float(T_initial["T_steel"])
    dr, r_in, r_out, r_mid, A_face, m_shell = prepare_mesh(cfg, mat, T_ref_mesh)
    inj_w = injection_weights(R_coil, cfg["d_out"], N, dr); d_coil_actual = 2.0 * R_coil
    if T_initial is None:
        H = np.full(N, float(np.interp(T_amb, mat["T_grid"], mat["H_grid"]))); T_steel = T_amb
    else:
        H = np.array(T_initial["H"], dtype=float); T_steel = float(T_initial["T_steel"])
    M_steel = steel_mass(cfg, 0.5 * (cfg["T_amb_low"] + cfg["T_water_in"]))
    A_outer = tank_outer_area(cfg); dt = cfg["dt_sec"]; n_steps = int(round(t_hr * 3600.0 / dt))
    rho_inlet = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    m_dot = cfg["flow_m3h"] * rho_inlet / 3600.0
    A_coil = np.pi * cfg["d_out"] * (2.0 * np.pi * R_coil * n_turns); dt_inv_m = dt / m_shell
    H_out, T_steel, peak_oil, Q_total, peak_Q = transient_kernel(H, T_steel, N, dr, dt_inv_m, A_face, inj_w, cfg["T_ms"], cfg["T_me"], mat["T_grid"], mat["H_grid"], mat["rho_grid"], mat["cp_sens_grid"], mat["k_grid"], mat["mu_grid"], cfg["T_water_arr"], cfg["rho_water_arr"], cfg["cp_water_arr"], cfg["mu_water_arr"], cfg["k_water_arr"], m_dot, cfg["T_water_in"], cfg["d_in"], cfg["d_out"], cfg["R_f_out"], A_coil, d_coil_actual, cfg["T_tube_arr"], cfg["k_tube_arr"], M_steel, cfg["T_wall_arr"], cfg["k_wall_arr"], cfg["cp_wall_arr"], A_outer, cfg["t_wall"], cfg["t_ins"], cfg["k_ins"], cfg["h_air"], T_amb, dt, n_steps, cfg["has_agitator"], cfg["ag_C"], cfg["ag_a"], cfg["N_rpm"], cfg["D_imp"], cfg["visc_corr"], cfg["beta_liq"], cfg["dT_natural"])
    return pack_result(cfg, mat, R_coil, n_turns, H_out, T_steel, m_shell, T_initial, peak_oil, Q_total, peak_Q)

def run_cool(cfg, mat, R_coil, n_turns, T_amb, T_initial, t_hr):
    if T_initial is None: raise ValueError("run_cool needs a start state.")
    N = cfg["N"]; T_ref_mesh = float(T_initial["T_steel"])
    dr, r_in, r_out, r_mid, A_face, m_shell = prepare_mesh(cfg, mat, T_ref_mesh)
    inj_w = injection_weights(R_coil, cfg["d_out"], N, dr)
    H = np.array(T_initial["H"], dtype=float); T_steel = float(T_initial["T_steel"])
    M_steel = steel_mass(cfg, 0.5 * (cfg["T_amb_low"] + cfg["T_water_in"]))
    A_outer = tank_outer_area(cfg); dt = cfg["dt_sec"]; n_steps = int(round(t_hr * 3600.0 / dt))
    L_coil = 2.0 * np.pi * R_coil * n_turns
    V_metal = np.pi * 0.25 * (cfg["d_out"] ** 2 - cfg["d_in"] ** 2) * L_coil
    V_water = np.pi * 0.25 * cfg["d_in"] ** 2 * L_coil
    rho_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["rho_tube_arr"]))
    cp_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["cp_tube_arr"]))
    rho_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    cp_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["cp_water_arr"]))
    M_coil_metal = rho_tube_ref * V_metal; M_coil_water = rho_water_ref * V_water
    cp_oil_ref = float(np.interp(cfg["T_water_in"], mat["T_grid"], mat["cp_sens_grid"]))
    extra_kg = (M_coil_metal * cp_tube_ref + M_coil_water * cp_water_ref) / cp_oil_ref
    m_eff = m_shell + extra_kg * inj_w; dt_inv_m_eff = dt / m_eff
    H_out, T_steel = cooldown_kernel(H, T_steel, N, dr, dt_inv_m_eff, A_face, mat["T_grid"], mat["H_grid"], mat["k_grid"], M_steel, cfg["T_wall_arr"], cfg["k_wall_arr"], cfg["cp_wall_arr"], A_outer, cfg["t_wall"], cfg["t_ins"], cfg["k_ins"], cfg["h_air"], T_amb, dt, n_steps)
    return pack_result(cfg, mat, R_coil, n_turns, H_out, T_steel, m_shell, T_initial, 0.0, 0.0, 0.0)

def _check(cancel):
    if cancel is not None and cancel.is_set(): raise Cancelled()

def default_workers():
    env = os.environ.get("VCH_WORKERS", "").strip()
    if env.isdigit() and int(env) >= 1: return int(env)
    n = os.cpu_count() or 1
    return max(1, min(8, n - 1 if n >= 4 else n))

def warm_up(cfg, mat):
    run_heat(cfg, mat, 0.5 * cfg["r_inner"], 3, cfg["T_amb_low"], None, 0.0)

def _cold_run(cfg, mat, task):
    idx, pname, Tamb, R, n, R_ratio = task
    r = run_heat(cfg, mat, R, n, Tamb, None, cfg["t_window_hr"])
    r["T_amb"] = Tamb; r["preset"] = pname; r["start_type"] = "cold"; r["R_ratio"] = R_ratio
    return idx, r

_W = {}
def _report(kind, **kw):
    q = _W.get("status_q")
    if q is None: return
    kw.update(kind=kind, pid=_W["pid"], t_wall=time.perf_counter(), t_cpu=time.process_time())
    try: q.put_nowait(kw)
    except Exception: pass

def _init_worker(cfg, mat, status_q=None):
    _W["cfg"], _W["mat"] = cfg, mat; _W["status_q"], _W["pid"] = status_q, os.getpid()
    _report("init"); warm_up(cfg, mat); _report("ready")

def _sweep_task(task):
    idx = task[0]; _report("start", idx=idx)
    try: return _cold_run(_W["cfg"], _W["mat"], task)
    finally: _report("end", idx=idx)

def _count(results): return sum(r is not None for r in results)

def _run_sequential(cfg, mat, tasks, results, tick, cancel):
    for t in tasks:
        _check(cancel)
        idx, r = _cold_run(cfg, mat, t); results[idx] = r; tick(_count(results))

def _run_threads(cfg, mat, tasks, results, tick, cancel, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_cold_run, cfg, mat, t) for t in tasks]
        try:
            for f in as_completed(futs):
                _check(cancel); idx, r = f.result(); results[idx] = r; tick(_count(results))
        except BaseException:
            for f in futs: f.cancel()
            raise

def _stop_pool(ex):
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try: p.terminate()
        except Exception: pass

def _run_processes(cfg, mat, tasks, results, tick, cancel, workers, monitor=None):
    ctx = mp.get_context("spawn"); status_q = ctx.Queue() if monitor is not None else None
    if monitor is not None: monitor.start(status_q)
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(cfg, mat, status_q))
    try:
        futs = [ex.submit(_sweep_task, t) for t in tasks]
        for f in as_completed(futs):
            _check(cancel); idx, r = f.result(); results[idx] = r; tick(_count(results))
    except BaseException:
        _stop_pool(ex); raise
    finally:
        ex.shutdown(wait=True)
        if monitor is not None: monitor.stop()
        if status_q is not None:
            status_q.close(); status_q.join_thread()

def run_cold_sweep(cfg, mat, geom, presets, progress=None, cancel=None, workers=None, mode="process", monitor=None):
    combos = [(pname, Tamb, g) for pname, Tamb in presets for g in geom]
    tasks = [(i, p, T, g["R"], g["n"], g["R_ratio"]) for i, (p, T, g) in enumerate(combos)]
    total = len(tasks); results = [None] * total; workers = max(1, min(workers or default_workers(), total))
    def tick(done, msg=""):
        if progress: progress(done, total, msg)
    _check(cancel); tick(0, "Preparing solver..."); warm_up(cfg, mat)
    if workers == 1 or mode == "sequential":
        tick(0, "Running sweep in process..."); _run_sequential(cfg, mat, tasks, results, tick, cancel); return results
    if mode == "process":
        tick(0, f"Starting {workers} workers...")
        try:
            _run_processes(cfg, mat, tasks, results, tick, cancel, workers, monitor=monitor); return results
        except Cancelled: raise
        except (BrokenExecutor, OSError, ImportError) as exc:
            tick(_count(results), f"Workers unavailable ({type(exc).__name__}); using threads...")
    pending = [t for t in tasks if results[t[0]] is None]
    _run_threads(cfg, mat, pending, results, tick, cancel, workers); return results

def run_hot(cfg, mat, R, n, presets, cool_states, progress=None, cancel=None):
    hot_rows = []; total = len(presets)
    for k, (pname, Tamb) in enumerate(presets):
        _check(cancel)
        r = run_heat(cfg, mat, R, n, Tamb, cool_states[pname], cfg["t_window_hr"])
        r["T_amb"] = Tamb; r["preset"] = pname; r["start_type"] = "hot"; r["R_ratio"] = R / cfg["r_inner"]
        hot_rows.append(r)
        if progress: progress(k + 1, total, f"hot run {pname}")
    return hot_rows

def run_cooldowns(cfg, mat, R, n, presets, cold_rows, progress=None, cancel=None):
    cold_at_selected = {}; cool_states = {}; cool_reports = {}; total = len(presets)
    for k, (pname, Tamb) in enumerate(presets):
        _check(cancel)
        cold_row = next(r for r in cold_rows if r["preset"] == pname and abs(r["R"] - R) < 1e-9 and r["n"] == n)
        cold_at_selected[pname] = cold_row
        T_ini = {"H": cold_row["H_final"], "T_steel": cold_row["T_steel"]}
        r = run_cool(cfg, mat, R, n, Tamb, T_ini, cfg["t_cooldown_hr"])
        cool_states[pname] = {"H": r["H_final"], "T_steel": r["T_steel"]}; cool_reports[pname] = r
        if progress: progress(k + 1, total, f"cooldown {pname}")
    return cold_at_selected, cool_states, cool_reports