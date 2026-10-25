"""Geometry: coil design grid (5 radii x whole-turn log points), radial mesh,
coil-heat injection weights and the coil / tank dimensions used by reports."""
import numpy as np


# ------------------------------------------------------------
# DESIGN GRID (phase A) AND SINGLE POINT (phase D)
# ------------------------------------------------------------
def whole_turns_logspace(n_min, n_max, npts):
    if n_max <= n_min:
        return np.array([n_min])
    if n_max - n_min + 1 <= npts:
        return np.arange(n_min, n_max + 1)
    raw = np.logspace(np.log10(n_min), np.log10(n_max), npts)
    turns = []
    for r in raw:
        v = max(n_min, min(n_max, int(round(r))))
        while v in turns and v < n_max:
            v += 1
        while v in turns and v > n_min:
            v -= 1
        turns.append(v)
    return np.array(sorted(set(turns)))


def max_turns(cfg):
    """Most whole turns that fit in the axial length (pitch = tube OD)."""
    return max(1, int(cfg["L_avail"] / cfg["d_out"]))


def build_geometry(cfg):
    r = cfg["r_inner"]; d_out = cfg["d_out"]
    radii = np.linspace(0.33 * r, 0.75 * r, 5)
    n_max = max_turns(cfg)
    turns = whole_turns_logspace(1, n_max, cfg["n_samples"])
    rows = []
    for R in radii:
        for n in turns:
            Lc = 2.0 * np.pi * R * n
            rows.append(dict(R=float(R), n=int(n), L=float(Lc),
                             A=np.pi * d_out * Lc, R_ratio=float(R / r)))
    return rows, n_max, radii, turns


def coil_properties(cfg, R, n):
    """Dimensions and masses of one selected coil (used in the report)."""
    L_coil = 2.0 * np.pi * R * n
    A_coil = np.pi * cfg["d_out"] * L_coil
    rho_tube_ref = float(np.interp(cfg["T_water_in"], cfg["T_tube_arr"], cfg["rho_tube_arr"]))
    M_metal = rho_tube_ref * np.pi * 0.25 * (cfg["d_out"] ** 2 - cfg["d_in"] ** 2) * L_coil
    rho_water_ref = float(np.interp(cfg["T_water_in"], cfg["T_water_arr"], cfg["rho_water_arr"]))
    V_water = np.pi * 0.25 * cfg["d_in"] ** 2 * L_coil
    M_water = rho_water_ref * V_water
    return dict(L_coil=L_coil, A_coil=A_coil, M_metal=M_metal,
                V_water=V_water, M_water=M_water,
                axial_height=n * cfg["d_out"])


def coil_index(cfg, R):
    """Radial cell that holds the coil."""
    return int(R / (cfg["r_inner"] / cfg["N"]))


# ------------------------------------------------------------
# RADIAL MESH AND TANK
# ------------------------------------------------------------
def prepare_mesh(cfg, mat, T_ref):
    N = cfg["N"]; r = cfg["r_inner"]; L = cfg["L_avail"]; dr = r / N
    idx = np.arange(N)
    r_in = idx * dr; r_out = (idx + 1) * dr; r_mid = 0.5 * (r_in + r_out)
    A_face = 2.0 * np.pi * r_mid[:-1] * L
    rho_ref = float(np.interp(T_ref, mat["T_grid"], mat["rho_grid"]))
    m_shell = np.pi * (r_out ** 2 - r_in ** 2) * L * rho_ref
    return dr, r_in, r_out, r_mid, A_face, m_shell


def injection_weights(R_coil, d_out, N, dr):
    band_lo = R_coil - d_out/2.0
    band_hi = R_coil + d_out/2.0
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
    
    # Calculate exact split based on tube surface areas
    area_in = R_coil - d_out / 2.0
    area_out = R_coil + d_out / 2.0
    frac_out = area_out / (area_in + area_out)
    frac_in = 1.0 - frac_out
    
    return frac_in * w_in + frac_out * w_out


def steel_mass(cfg, T_ref):
    r = cfg["r_inner"]; t = cfg["t_wall"]; L = cfg["L_avail"]
    V = np.pi * ((r + t) ** 2 - r ** 2) * L
    rho_ref = float(np.interp(T_ref, cfg["T_wall_arr"], cfg["rho_wall_arr"]))
    return rho_ref * V


def tank_outer_area(cfg):
    Ro = cfg["r_inner"] + cfg["t_wall"]
    return 2.0 * np.pi * Ro * cfg["L_avail"] + 2.0 * np.pi * Ro ** 2
