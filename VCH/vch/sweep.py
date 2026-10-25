"""Sweep: Numba heating kernel, single heating run, phase-A cold sweep and
phase-D hot run.

Phase A is a multi-core sweep: every simulation is its own task, the tasks go
into one queue, and a pool of worker *processes* (each with its own interpreter
and GIL) takes the next free task.  Results come back through `as_completed`, so
progress is counted in the calling process with no locks or shared memory."""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures import BrokenExecutor

import numpy as np

from .physics import _jit, calc_Q, pack_result
from .geometry import (prepare_mesh, injection_weights, steel_mass, tank_outer_area)


class Cancelled(Exception):
    """Raised when the user cancels a running phase."""


# ------------------------------------------------------------
# KERNEL
# ------------------------------------------------------------
@_jit
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
    peak_oil = -1e9
    Q_total = 0.0
    peak_Q = 0.0
    T = np.empty(N)
    k_arr = np.empty(N)

    for i in range(N):
        Ti = np.interp(H[i], H_grid, T_grid)
        if Ti > peak_oil:
            peak_oil = Ti

    for step in range(n_steps):
        for i in range(N):
            T[i] = np.interp(H[i], H_grid, T_grid)
            if T[i] > peak_oil:
                peak_oil = T[i]

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
        if step_Q > peak_Q:
            peak_Q = step_Q

        for i in range(N - 1):
            kavg = 0.5 * (k_arr[i] + k_arr[i + 1])
            Qc = kavg * A_face[i] * (T[i] - T[i + 1]) / dr
            H[i] -= Qc * dt_inv_m[i]
            H[i + 1] += Qc * dt_inv_m[i + 1]

        Q_os = k_arr[N - 1] * A_tank_outer * (T[N - 1] - T_steel) / (dr * 0.5)
        H[N - 1] -= Q_os * dt_inv_m[N - 1]

        k_wall_now = np.interp(T_steel, T_wall_arr, k_wall_arr)
        cp_wall_now = np.interp(T_steel, T_wall_arr, cp_wall_arr)
        R_conv = 1.0 / h_air
        R_wall_half = (t_wall / 2.0) / k_wall_now
        R_ins = (t_ins / k_ins) if t_ins > 0.0 else 0.0
        U_air = 1.0 / (R_wall_half + R_ins + R_conv)
        Q_sa = U_air * A_tank_outer * (T_steel - T_amb)
        T_steel += (Q_os - Q_sa) * dt / (M_steel * cp_wall_now)

    for i in range(N):
        Ti = np.interp(H[i], H_grid, T_grid)
        if Ti > peak_oil:
            peak_oil = Ti

    return H, T_steel, peak_oil, Q_total, peak_Q


# ------------------------------------------------------------
# ONE HEATING RUN
# ------------------------------------------------------------
def run_heat(cfg, mat, R_coil, n_turns, T_amb, T_initial, t_hr):
    """Heat the tank for t_hr hours.  T_initial=None means a cold start from
    ambient; otherwise a dict with the residual state {"H", "T_steel"}."""
    N = cfg["N"]
    T_ref_mesh = T_amb if T_initial is None else float(T_initial["T_steel"])
    dr, r_in, r_out, r_mid, A_face, m_shell = prepare_mesh(cfg, mat, T_ref_mesh)
    inj_w = injection_weights(R_coil, cfg["d_out"], N, dr)
    d_coil_actual = 2.0 * R_coil

    if T_initial is None:
        H = np.full(N, float(np.interp(T_amb, mat["T_grid"], mat["H_grid"])))
        T_steel = T_amb
    else:
        H = np.array(T_initial["H"], dtype=float)
        T_steel = float(T_initial["T_steel"])

    M_steel = steel_mass(cfg, 0.5 * (cfg["T_amb_low"] + cfg["T_water_in"]))
    A_outer = tank_outer_area(cfg)
    dt = cfg["dt_sec"]
    n_steps = int(round(t_hr * 3600.0 / dt))

    rho_inlet = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    m_dot = cfg["flow_m3h"] * rho_inlet / 3600.0
    A_coil = np.pi * cfg["d_out"] * (2.0 * np.pi * R_coil * n_turns)
    dt_inv_m = dt / m_shell

    H_out, T_steel, peak_oil, Q_total, peak_Q = transient_kernel(
        H, T_steel, N, dr, dt_inv_m, A_face, inj_w,
        cfg["T_ms"], cfg["T_me"],
        mat["T_grid"], mat["H_grid"], mat["rho_grid"], mat["cp_sens_grid"], mat["k_grid"], mat["mu_grid"],
        cfg["T_water_arr"], cfg["rho_water_arr"], cfg["cp_water_arr"],
        cfg["mu_water_arr"], cfg["k_water_arr"],
        m_dot, cfg["T_water_in"],
        cfg["d_in"], cfg["d_out"], cfg["R_f_out"], A_coil, d_coil_actual,
        cfg["T_tube_arr"], cfg["k_tube_arr"],
        M_steel,
        cfg["T_wall_arr"], cfg["k_wall_arr"], cfg["cp_wall_arr"],
        A_outer, cfg["t_wall"], cfg["t_ins"], cfg["k_ins"], cfg["h_air"],
        T_amb, dt, n_steps,
        cfg["has_agitator"], cfg["ag_type"], cfg["N_rpm"], cfg["D_imp"], cfg["visc_corr"],
        cfg["beta_liq"], cfg["dT_natural"])

    return pack_result(cfg, mat, R_coil, n_turns, H_out, T_steel, m_shell,
                       T_initial, peak_oil, Q_total, peak_Q)


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# ------------------------------------------------------------
# PHASE A: COLD SWEEP (all designs x 3 ambients, multi-core)
# ------------------------------------------------------------
def default_workers():
    """Worker processes to use: all cores (max 8), leaving one free for the window
    on machines with 4+ cores.  Set the environment variable VCH_WORKERS to override
    (VCH_WORKERS=1 runs everything in-process)."""
    env = os.environ.get("VCH_WORKERS", "").strip()
    if env.isdigit() and int(env) >= 1:
        return int(env)
    n = os.cpu_count() or 1
    return max(1, min(8, n - 1 if n >= 4 else n))


def warm_up(cfg, mat):
    """Zero-length heating run.  The first time on a machine it compiles the Numba
    kernel and writes it to the on-disk cache; afterwards it just loads it.  Run in
    the calling process *before* the workers start, so they never compile (several
    processes compiling at once can deadlock on the cache files)."""
    run_heat(cfg, mat, 0.5 * cfg["r_inner"], 3, cfg["T_amb_low"], None, 0.0)


def _cold_run(cfg, mat, task):
    """One simulation = one task: (index, preset, T_amb, R, n, R_ratio)."""
    idx, pname, Tamb, R, n, R_ratio = task
    r = run_heat(cfg, mat, R, n, Tamb, None, cfg["t_window_hr"])
    r["T_amb"] = Tamb; r["preset"] = pname; r["start_type"] = "cold"
    r["R_ratio"] = R_ratio
    return idx, r


# ---- code that runs inside each worker process ----------------------------
_W = {}


def _init_worker(cfg, mat):
    _W["cfg"], _W["mat"] = cfg, mat
    warm_up(cfg, mat)  # loads the compiled kernel from the cache (no compiling)


def _sweep_task(task):
    return _cold_run(_W["cfg"], _W["mat"], task)


# ---- the three ways to execute the task list --------------------------------
def _count(results):
    return sum(r is not None for r in results)


def _run_sequential(cfg, mat, tasks, results, tick, cancel):
    for t in tasks:
        _check(cancel)
        idx, r = _cold_run(cfg, mat, t)
        results[idx] = r
        tick(_count(results))


def _run_threads(cfg, mat, tasks, results, tick, cancel, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_cold_run, cfg, mat, t) for t in tasks]
        try:
            for f in as_completed(futs):
                _check(cancel)
                idx, r = f.result()
                results[idx] = r
                tick(_count(results))
        except BaseException:
            for f in futs:
                f.cancel()
            raise


def _stop_pool(ex):
    """Drop queued tasks and stop the workers now (cancel / error).  The worker
    handles are taken first: shutdown() clears the executor's process table."""
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass


def _run_processes(cfg, mat, tasks, results, tick, cancel, workers):
    # "spawn" everywhere: the same start method Windows uses, and safe next to Qt threads
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(cfg, mat))
    try:
        futs = [ex.submit(_sweep_task, t) for t in tasks]      # whole queue at once
        for f in as_completed(futs):                           # each result the moment it ends
            _check(cancel)
            idx, r = f.result()
            results[idx] = r
            tick(_count(results))
    except BaseException:
        _stop_pool(ex)
        raise
    ex.shutdown(wait=True)


def run_cold_sweep(cfg, mat, geom, presets, progress=None, cancel=None,
                   workers=None, mode="process"):
    """Heat every (R, n) design from a cold start at every ambient.
    Returns rows in P6 order: preset-major, then design order.
    mode: "process" (default; falls back to threads if processes cannot start),
          "thread", or "sequential"."""
    combos = [(pname, Tamb, g) for pname, Tamb in presets for g in geom]
    tasks = [(i, p, T, g["R"], g["n"], g["R_ratio"]) for i, (p, T, g) in enumerate(combos)]
    total = len(tasks)
    results = [None] * total
    workers = max(1, min(workers or default_workers(), total))

    def tick(done, msg=""):
        if progress:
            progress(done, total, msg)

    _check(cancel)
    tick(0, "Preparing the solver (compiles on first use)...")
    warm_up(cfg, mat)

    if workers == 1 or mode == "sequential":
        tick(0, "Running the sweep in this process...")
        _run_sequential(cfg, mat, tasks, results, tick, cancel)
        return results

    if mode == "process":
        tick(0, f"Starting {workers} worker processes...")
        try:
            _run_processes(cfg, mat, tasks, results, tick, cancel, workers)
            return results
        except Cancelled:
            raise
        except (BrokenExecutor, OSError, ImportError) as exc:
            tick(_count(results), f"Worker processes unavailable ({type(exc).__name__}); "
                                  f"continuing with threads...")

    pending = [t for t in tasks if results[t[0]] is None]
    _run_threads(cfg, mat, pending, results, tick, cancel, workers)
    return results


# ------------------------------------------------------------
# PHASE D: HOT RUN (selected coil, from the cooldown residual state)
# ------------------------------------------------------------
def run_hot(cfg, mat, R, n, presets, cool_states, progress=None, cancel=None):
    hot_rows = []
    total = len(presets)
    for k, (pname, Tamb) in enumerate(presets):
        _check(cancel)
        r = run_heat(cfg, mat, R, n, Tamb, cool_states[pname], cfg["t_window_hr"])
        r["T_amb"] = Tamb; r["preset"] = pname; r["start_type"] = "hot"
        r["R_ratio"] = R / cfg["r_inner"]
        hot_rows.append(r)
        if progress:
            progress(k + 1, total, f"hot run {pname}")
    return hot_rows
