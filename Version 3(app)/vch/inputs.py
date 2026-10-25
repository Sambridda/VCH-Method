"""Inputs: every value the study needs, grouped, validated and echoed back.

Part 1 (pure Python) holds the defaults, the 5-point property tables, the
validation rules, the config builder and the echo summary.
Part 2 (PySide6) holds the input form that replaces the ~30 terminal prompts.
"""
import json
import os

import numpy as np

from .physics import WATER_T, WATER_RHO, WATER_CP, WATER_MU, WATER_K
from .geometry import max_turns

# ============================================================
# PART 1 - DATA MODEL
# ============================================================
ENV_PRESETS = [("Indoor  (h_air = 5 W/m2K)", 5.0),
               ("Outdoor  (h_air = 10 W/m2K)", 10.0),
               ("Outdoor windy  (h_air = 25 W/m2K)", 25.0)]

AGITATORS = [
    (1, "Flat-Blade Turbine (Pitched/Radial) - Turbulent, C=0.60, a=0.67"),
    (2, "Standard Paddle Agitator - Transitional, C=0.54, a=0.67"),
    (3, "Anchor Agitator (Swept-wall) - Laminar, C=0.40, a=0.50"),
    (4, "Helical Ribbon / Screw - Extreme Viscosity, C=0.35, a=0.50"),
]

# (key, label, unit) for every 5-point property table
PROP_SPECS = {
    "water":  [("rho", "Density", "kg/m3"), ("cp", "Specific heat", "J/(kg*K)"),
               ("mu", "Viscosity", "Pa*s"), ("k", "Thermal cond.", "W/(m*K)")],
    "solid":  [("rho", "Density", "kg/m3"), ("cp", "Specific heat", "J/(kg*K)"),
               ("k", "Thermal cond.", "W/(m*K)"), ("mu", "Viscosity", "Pa*s")],
    "liquid": [("rho", "Density", "kg/m3"), ("cp", "Specific heat", "J/(kg*K)"),
               ("k", "Thermal cond.", "W/(m*K)"), ("mu", "Viscosity", "Pa*s")],
    "wall":   [("rho", "Density", "kg/m3"), ("cp", "Specific heat", "J/(kg*K)"),
               ("k", "Thermal cond.", "W/(m*K)")],
    "tube":   [("rho", "Density", "kg/m3"), ("cp", "Specific heat", "J/(kg*K)"),
               ("k", "Thermal cond.", "W/(m*K)")],
}
TABLE_KINDS = list(PROP_SPECS)
TABLE_TITLES = {"water": "Coil fluid", "solid": "Tank fluid - solid", "liquid": "Tank fluid - liquid",
                "wall": "Tank wall", "tube": "Coil tube"}

SCALAR_DEFAULTS = dict(
    T_amb_low=2.0, T_amb_mid=13.0, T_amb_hot=22.0,
    T_water_in=70.0, flow_m3h=1.17, beta_liq=7e-4, dT_natural=5.0,
    has_agitator=False, ag_type=1, N_rpm=60.0, D_imp=0.0, visc_corr=1.0,
    fluid_name="water", T_water_low=2.0,
    tank_name="oil", T_ms=31.0, T_me=41.0, Hfus=65000.0,
    wall_name="steel", t_wall=0.005,
    coil_mat="steel", d_in=0.0409, d_out=0.04826, R_f_out=0.0004,
    env_index=0, t_ins=0.0, k_ins=0.04,
    r_inner=1.22, L_avail=2.3, fill_from_geometry=True, M_fluid=0.0,
    t_window_hr=12.0, t_cooldown_hr=12.0, N=100, dt_sec=1.0, n_samples=10,
)


def table_temps(inp):
    """Temperature points (5 each) the property tables are defined at."""
    return dict(
        water=np.linspace(inp["T_water_low"], inp["T_water_in"], 5),
        solid=np.linspace(inp["T_amb_low"], inp["T_ms"], 5),
        liquid=np.linspace(inp["T_me"], inp["T_water_in"], 5),
        wall=np.linspace(inp["T_amb_low"], inp["T_water_in"], 5),
        tube=np.linspace(inp["T_amb_low"], inp["T_water_in"], 5),
    )


def default_table(kind, T_arr, fluid_name="water"):
    """Default property values for one table, or None when there are none
    (a coil fluid other than water has to be typed in)."""
    T = np.asarray(T_arr, dtype=float)
    if kind == "water":
        if "water" not in str(fluid_name).lower():
            return None
        return dict(rho=np.interp(T, WATER_T, WATER_RHO).tolist(),
                    cp=np.interp(T, WATER_T, WATER_CP).tolist(),
                    mu=np.interp(T, WATER_T, WATER_MU).tolist(),
                    k=np.interp(T, WATER_T, WATER_K).tolist())
    if kind == "solid":
        return dict(rho=[920.0] * 5, cp=[1600.0] * 5, k=[0.180] * 5, mu=[1e5] * 5)
    if kind == "liquid":
        return dict(rho=[902.5 - 0.53 * t for t in T],
                    cp=[1750.0 + 2.9 * t for t in T],
                    k=[0.85] * 5,
                    mu=[1.85e-6 * float(np.exp(3080.0 / (t + 273.15))) for t in T])
    if kind in ("wall", "tube"):
        return dict(rho=[7850.0] * 5, cp=[500.0] * 5, k=[16.0] * 5)
    raise KeyError(kind)


def default_inputs():
    inp = dict(SCALAR_DEFAULTS)
    temps = table_temps(inp)
    inp["tables"] = {k: default_table(k, temps[k], inp["fluid_name"]) for k in TABLE_KINDS}
    return inp


def _arr(table, key):
    vals = table.get(key) if table else None
    if vals is None or len(vals) != 5 or any(v is None or not np.isfinite(v) for v in vals):
        raise ValueError
    return np.array(vals, dtype=float)


def build_cfg(inp):
    """Raw inputs -> the cfg dictionary the engine uses (same keys as P6).
    Returns (cfg, errors, warnings); cfg is None when there are errors."""
    errors, warnings = [], []
    g = inp
    temps = table_temps(g)

    # ---- scalar sanity ------------------------------------------------
    def need(cond, msg):
        if not cond:
            errors.append(msg)

    need(g["T_water_in"] > g["T_amb_low"], "Coil inlet temperature must be above the low ambient.")
    need(g["flow_m3h"] > 0, "Coil flow must be > 0.")
    need(g["T_ms"] < g["T_me"], "Melt-start temperature must be below melt-end temperature.")
    need(g["Hfus"] > 0, "Latent heat of fusion must be > 0.")
    need(g["T_amb_low"] < g["T_ms"],
         "Low ambient must be below the melt-start temperature (cold start must begin solid).")
    need(g["T_me"] < g["T_water_in"],
         "Melt-end temperature must be below the coil inlet temperature.")
    need(-50.0 <= g["T_water_low"] <= g["T_water_in"],
         "Lowest coil-fluid temperature must be between -50 degC and the coil inlet temperature.")
    need(g["t_wall"] > 0, "Wall thickness must be > 0.")
    need(g["d_in"] > 0 and g["d_out"] > g["d_in"], "Tube: need 0 < inner diameter < outer diameter.")
    need(g["R_f_out"] >= 0, "Fouling resistance must be >= 0.")
    need(g["t_ins"] >= 0, "Insulation thickness must be >= 0.")
    if g["t_ins"] > 0:
        need(g["k_ins"] > 0, "Insulation conductivity must be > 0 when insulation is used.")
    need(g["r_inner"] > 0, "Tank inner radius must be > 0.")
    need(g["L_avail"] > 0, "Axial length must be > 0.")
    need(g["t_window_hr"] > 0, "Heating window must be > 0.")
    need(g["t_cooldown_hr"] > 0, "Cooldown window must be > 0.")
    need(20 <= g["N"] <= 500, "Radial nodes N must be between 20 and 500.")
    need(0.1 <= g["dt_sec"] <= 60.0, "Timestep dt must be between 0.1 and 60 s.")
    if g["has_agitator"]:
        need(g["N_rpm"] > 0, "Agitator speed must be > 0.")
        need(g["D_imp"] >= 0, "Impeller diameter must be >= 0 (0 = tank radius).")
        need(g["visc_corr"] > 0, "Viscosity correction factor must be > 0.")
    else:
        need(g["beta_liq"] > 0, "Thermal expansion coefficient must be > 0 (no agitator).")
        need(g["dT_natural"] > 0, "Natural-convection delta-T must be > 0 (no agitator).")

    # ---- 5-point property tables ----------------------------------------
    arrays = {}
    for kind, spec in PROP_SPECS.items():
        for key, label, unit in spec:
            try:
                a = _arr(g["tables"].get(kind), key)
            except ValueError:
                errors.append(f"{TABLE_TITLES[kind]} table: '{label}' needs 5 numeric values.")
                continue
            if np.any(a <= 0):
                errors.append(f"{TABLE_TITLES[kind]} table: '{label}' must be positive.")
            arrays[(kind, key)] = a

    if errors:
        return None, errors, warnings

    # ---- coil-fluid temperature coverage (P6 check) -----------------------
    T_eff_min = 0.5 * (g["T_water_in"] + g["T_amb_low"])
    if T_eff_min < g["T_water_low"]:
        errors.append(
            f"Coil-fluid temperature would fall to ({g['T_water_in']:.1f}+{g['T_amb_low']:.1f})/2 = "
            f"{T_eff_min:.1f} degC, below your property table's lowest point "
            f"({g['T_water_low']:.1f} degC). Lower 'Lowest coil-fluid temperature'.")
        return None, errors, warnings

    g_n_max = max_turns({"L_avail": g["L_avail"], "d_out": g["d_out"]})
    need(2 <= g["n_samples"] <= max(2, g_n_max),
         f"Turn samples must be between 2 and {max(2, g_n_max)}.")
    if errors:
        return None, errors, warnings

    D_imp = g["D_imp"]
    if g["has_agitator"] and D_imp <= 0.0:
        D_imp = g["r_inner"]

    h_air = ENV_PRESETS[int(g["env_index"])][1]

    cfg = dict(
        fluid_name=g["fluid_name"] or "water",
        T_water_arr=temps["water"],
        rho_water_arr=arrays[("water", "rho")], cp_water_arr=arrays[("water", "cp")],
        mu_water_arr=arrays[("water", "mu")], k_water_arr=arrays[("water", "k")],
        T_water_low=g["T_water_low"],
        tank_name=g["tank_name"] or "oil", T_ms=g["T_ms"], T_me=g["T_me"], Hfus=g["Hfus"],
        T_solid_arr=temps["solid"],
        rho_solid_arr=arrays[("solid", "rho")], cp_solid_arr=arrays[("solid", "cp")],
        k_solid_arr=arrays[("solid", "k")], mu_solid_arr=arrays[("solid", "mu")],
        T_liquid_arr=temps["liquid"],
        rho_liquid_arr=arrays[("liquid", "rho")], cp_liquid_arr=arrays[("liquid", "cp")],
        k_liquid_arr=arrays[("liquid", "k")], mu_liquid_arr=arrays[("liquid", "mu")],
        wall_name=g["wall_name"] or "steel", t_wall=g["t_wall"],
        T_wall_arr=temps["wall"],
        rho_wall_arr=arrays[("wall", "rho")], cp_wall_arr=arrays[("wall", "cp")],
        k_wall_arr=arrays[("wall", "k")],
        coil_mat=g["coil_mat"] or "steel", d_in=g["d_in"], d_out=g["d_out"], R_f_out=g["R_f_out"],
        T_tube_arr=temps["tube"],
        rho_tube_arr=arrays[("tube", "rho")], cp_tube_arr=arrays[("tube", "cp")],
        k_tube_arr=arrays[("tube", "k")],
        h_air=h_air, t_ins=g["t_ins"], k_ins=g["k_ins"],
        r_inner=g["r_inner"], L_avail=g["L_avail"],
        T_water_in=g["T_water_in"], flow_m3h=g["flow_m3h"],
        T_amb_low=g["T_amb_low"], T_amb_mid=g["T_amb_mid"], T_amb_hot=g["T_amb_hot"],
        t_window_hr=g["t_window_hr"], t_cooldown_hr=g["t_cooldown_hr"],
        dt_sec=float(g["dt_sec"]), N=int(g["N"]), n_samples=int(g["n_samples"]),
        has_agitator=1 if g["has_agitator"] else 0,
        ag_type=int(g["ag_type"]) if g["has_agitator"] else 0,
        N_rpm=float(g["N_rpm"]) if g["has_agitator"] else 0.0,
        D_imp=float(D_imp) if g["has_agitator"] else 0.0,
        visc_corr=float(g["visc_corr"]) if g["has_agitator"] else 1.0,
        beta_liq=g["beta_liq"], dT_natural=g["dT_natural"],
    )

    # ---- fluid mass (display value; the simulation uses tank geometry x density)
    V_tank = np.pi * cfg["r_inner"] ** 2 * cfg["L_avail"]
    m_fill = V_tank * float(np.interp(cfg["T_amb_low"], cfg["T_solid_arr"], cfg["rho_solid_arr"]))
    cfg["M_fill"] = m_fill
    cfg["M_fluid"] = m_fill if g["fill_from_geometry"] else float(g["M_fluid"])
    if not g["fill_from_geometry"]:
        if cfg["M_fluid"] <= 0:
            errors.append("Fluid mass must be > 0 (or tick 'fill from geometry').")
            return None, errors, warnings
        if abs(cfg["M_fluid"] - m_fill) / m_fill > 0.10:
            warnings.append(
                f"Fluid mass {cfg['M_fluid']:.0f} kg is more than 10% off the geometry fill "
                f"({m_fill:.0f} kg). The simulation always uses the geometry fill; the mass you "
                f"enter is only shown in the report.")

    # ---- advisory warnings ---------------------------------------------------
    if cfg["n_samples"] < 5:
        warnings.append(f"Only {cfg['n_samples']} turn samples - the sweep curves will be blocky.")
    if cfg["N"] < 50:
        kg = m_fill / cfg["N"]
        warnings.append(f"Coarse mesh (N={cfg['N']}): each cell holds about {kg:.0f} kg, so liquid mass "
                        f"changes in whole-cell steps. Use N >= 100 for smoother results.")
    if cfg["dt_sec"] > 10:
        warnings.append(f"Large timestep (dt={cfg['dt_sec']} s) - accuracy drops above about 10 s.")
    return cfg, errors, warnings


def estimate_runtime(cfg):
    """Rough run counts and minutes (P6 estimate; the progress ETA self-corrects)."""
    n_max = max_turns(cfg)
    n_pts = min(cfg["n_samples"], n_max)
    est_runs = 5 * n_pts * 3
    per_run_s = 3.0 * (cfg["N"] / 100.0) * (1.0 / cfg["dt_sec"])
    cold = est_runs * per_run_s / 60.0
    cool = 3 * per_run_s * (cfg["t_cooldown_hr"] / cfg["t_window_hr"]) / 60.0
    hot = 3 * per_run_s / 60.0
    return dict(runs=est_runs, n_max=n_max, cold_min=cold, cool_min=cool, hot_min=hot,
                total_min=cold + cool + hot)


def summary_text(cfg, warnings=()):
    """Echo summary of the inputs (the P6 SUMMARY block plus run estimates)."""
    e = estimate_runtime(cfg)
    L = []
    L.append("SUMMARY")
    L.append("=" * 64)
    L.append(f"Coil fluid  : {cfg['fluid_name']}  band {cfg['T_water_arr'][0]:.1f}..{cfg['T_water_arr'][-1]:.1f} degC")
    L.append(f"Tank fluid  : {cfg['tank_name']}  melt {cfg['T_ms']}-{cfg['T_me']} degC  "
             f"Hf={cfg['Hfus']:.0f} J/kg  M={cfg['M_fluid']:.0f} kg")
    L.append(f"Wall        : {cfg['wall_name']}  t={cfg['t_wall']} m")
    L.append(f"Coil tube   : {cfg['coil_mat']}  d_in={cfg['d_in']}  d_out={cfg['d_out']}  Rf={cfg['R_f_out']}")
    L.append(f"Insulation  : t={cfg['t_ins']} m  k={cfg['k_ins']}  h_air={cfg['h_air']}")
    L.append(f"Tank        : r={cfg['r_inner']} m  L_avail={cfg['L_avail']} m")
    if cfg["has_agitator"]:
        L.append(f"Agitator    : Type {cfg['ag_type']}  N={cfg['N_rpm']} RPM  "
                 f"D={cfg['D_imp']:.2f} m  visc_corr={cfg['visc_corr']}")
    else:
        L.append("Agitator    : None (natural convection)")
    L.append(f"Water       : T_in={cfg['T_water_in']} degC  flow={cfg['flow_m3h']} m3/h")
    L.append(f"Ambients    : {cfg['T_amb_low']}/{cfg['T_amb_mid']}/{cfg['T_amb_hot']} degC")
    L.append(f"Windows     : heating {cfg['t_window_hr']:.1f} h, cooldown {cfg['t_cooldown_hr']:.1f} h")
    L.append(f"Settings    : N={cfg['N']}  dt={cfg['dt_sec']} s  n_samples={cfg['n_samples']}")
    L.append("")
    L.append(f"Coil turns range : 1 to {e['n_max']}  (= L_avail / d_out)")
    L.append(f"Cold sweep       : {e['runs']} runs  (~{e['cold_min']:.1f} min, single core)")
    L.append(f"Cooldowns        : 3 runs x {cfg['t_cooldown_hr']:.1f} h  (~{e['cool_min']:.1f} min)")
    L.append(f"Hot runs         : 3 runs x {cfg['t_window_hr']:.1f} h  (~{e['hot_min']:.1f} min)")
    L.append(f"Total estimate   : ~{e['total_min']:.1f} min (the sweep runs on several cores)")
    if warnings:
        L.append("")
        L.append("WARNINGS")
        L.append("-" * 64)
        for w in warnings:
            L.append(f"* {w}")
    return "\n".join(L)


# ---- save / load -----------------------------------------------------------
def save_inputs(path, inp):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inp, f, indent=2)


def load_inputs(path):
    """Load a saved inputs file, filling anything missing with the defaults."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    inp = default_inputs()
    for k in SCALAR_DEFAULTS:
        if k in data:
            inp[k] = data[k]
    for kind in TABLE_KINDS:
        t = (data.get("tables") or {}).get(kind)
        if t:
            inp["tables"][kind] = t
    return inp


# ============================================================
# PART 2 - PYSIDE6 INPUT FORM
# ============================================================
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt, Signal  # noqa: E402

from .geometry import whole_turns_logspace  # noqa: E402

TEMP_KEYS = ("T_amb_low", "T_water_in", "T_water_low", "T_ms", "T_me", "fluid_name")
INFO_KEYS = ("r_inner", "L_avail", "d_out", "n_samples", "N", "dt_sec", "t_window_hr", "t_cooldown_hr")


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.10g}"


class NumberEdit(QtWidgets.QLineEdit):
    """Line edit that only accepts numbers (scientific notation allowed)."""

    def __init__(self, value, integer=False, width=120):
        super().__init__(_fmt(int(value) if integer else value))
        self.integer = integer
        self.setLocale(QtCore.QLocale.c())
        if integer:
            v = QtGui.QIntValidator(self)
        else:
            v = QtGui.QDoubleValidator(self)
            v.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
            v.setDecimals(15)
        v.setLocale(QtCore.QLocale.c())
        self.setValidator(v)
        self.setMaximumWidth(width)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def value(self):
        t = self.text().strip()
        return int(t) if self.integer else float(t)  # ValueError when blank / partial

    def set_value(self, v):
        self.setText(_fmt(int(v) if self.integer else v))


class PropertyTable(QtWidgets.QTableWidget):
    """5-point property table (temperature rows x property columns).  Follows
    the defaults until the user edits a cell, then keeps the user's numbers."""

    def __init__(self, kind):
        spec = PROP_SPECS[kind]
        super().__init__(5, 1 + len(spec))
        self.kind = kind
        self.keys = [k for k, _, _ in spec]
        self.dirty = False
        self._busy = False
        self.setHorizontalHeaderLabels(["T (degC)"] + [f"{lab}\n{unit}" for _, lab, unit in spec])
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setFixedHeight(44)
        self.setFixedHeight(44 + 5 * 26 + 4)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.itemChanged.connect(self._on_changed)
        self.set_temps(np.zeros(5))

    def _on_changed(self, item):
        if not self._busy and item.column() > 0:
            self.dirty = True

    def set_temps(self, T):
        self._busy = True
        for r in range(5):
            it = QtWidgets.QTableWidgetItem(f"{float(T[r]):.2f}")
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it.setBackground(QtGui.QColor(235, 238, 242))
            it.setForeground(QtGui.QColor(60, 60, 60))
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(r, 0, it)
        self._busy = False

    def set_values(self, table, dirty):
        self._busy = True
        for r in range(5):
            for c, key in enumerate(self.keys, start=1):
                exact = None if table is None else float(table[key][r])
                txt = "" if exact is None else _fmt(exact)
                it = QtWidgets.QTableWidgetItem(txt)
                # keep the exact value: the text only shows 10 digits, so it is
                # re-parsed only if the user edits the cell
                it.setData(Qt.ItemDataRole.UserRole, (txt, exact))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setItem(r, c, it)
        self._busy = False
        self.dirty = dirty

    def apply_defaults(self, T, fluid_name="water"):
        self.set_values(default_table(self.kind, T, fluid_name), dirty=False)

    def values(self):
        out = {}
        for c, key in enumerate(self.keys, start=1):
            col = []
            for r in range(5):
                it = self.item(r, c)
                try:
                    kept = it.data(Qt.ItemDataRole.UserRole)
                    if kept and kept[1] is not None and kept[0] == it.text():
                        col.append(kept[1])
                    else:
                        col.append(float(it.text()))
                except (ValueError, AttributeError):
                    col.append(None)
            out[key] = col
        return out


class ReviewDialog(QtWidgets.QDialog):
    """Echo summary of the inputs, shown before the study starts."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review inputs")
        self.resize(760, 640)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Check the summary below, then start Phase A (cold sweep)."))
        box = QtWidgets.QPlainTextEdit(text)
        box.setReadOnly(True)
        box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(box)
        bb = QtWidgets.QDialogButtonBox()
        go = bb.addButton("Start Phase A", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        go.setDefault(True)
        bb.addButton("Back to inputs", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class InputsForm(QtWidgets.QWidget):
    """All study inputs in grouped tabs.  Emits `confirmed(cfg, inputs)` once the
    values are valid and the user has accepted the echo summary."""

    confirmed = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fields = {}
        self.labels = {}
        self.tables = {}
        self._tw_low_manual = False
        self._loading = False
        d = default_inputs()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._page(self._tab_operating(d)), "Operating")
        self.tabs.addTab(self._page(self._tab_agitator(d)), "Agitator")
        self.tabs.addTab(self._page(self._tab_coil_fluid(d)), "Coil fluid")
        self.tabs.addTab(self._page(self._tab_tank_fluid(d)), "Tank fluid (PCM)")
        self.tabs.addTab(self._page(self._tab_hardware(d)), "Wall && coil tube")
        self.tabs.addTab(self._page(self._tab_simulation(d)), "Tank && simulation")

        bar = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load inputs...")
        b_save = QtWidgets.QPushButton("Save inputs...")
        b_reset = QtWidgets.QPushButton("Reset to defaults")
        b_load.clicked.connect(self.load_dialog)
        b_save.clicked.connect(self.save_dialog)
        b_reset.clicked.connect(lambda: self.set_inputs(default_inputs()))
        self.btn_go = QtWidgets.QPushButton("Review && start Phase A  \u25B6")
        self.btn_go.setObjectName("primary")
        self.btn_go.setMinimumHeight(34)
        self.btn_go.clicked.connect(self.review)
        for b in (b_load, b_save, b_reset):
            bar.addWidget(b)
        bar.addStretch()
        bar.addWidget(self.btn_go)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.tabs)
        root.addLayout(bar)

        for k in TEMP_KEYS:
            self.fields[k].textChanged.connect(self._on_temps_changed)
        for k in INFO_KEYS:
            self.fields[k].textChanged.connect(self._update_info)
        self.fields["T_amb_low"].textChanged.connect(self._follow_low_ambient)
        self.fields["T_water_low"].textEdited.connect(lambda _t: setattr(self, "_tw_low_manual", True))
        self.cb_agitator.toggled.connect(self._agitator_toggled)
        self.cb_fill.toggled.connect(self._fill_toggled)
        self.refresh_tables(reset=True)
        self._agitator_toggled(self.cb_agitator.isChecked())
        self._fill_toggled(self.cb_fill.isChecked())
        self._update_info()

    # ---- layout helpers ----------------------------------------------------
    @staticmethod
    def _page(widget):
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sc.setWidget(widget)
        return sc

    @staticmethod
    def _group(title, layout):
        g = QtWidgets.QGroupBox(title)
        g.setLayout(layout)
        return g

    def _num(self, form, key, label, value, unit="", integer=False, tip=""):
        w = NumberEdit(value, integer)
        if tip:
            w.setToolTip(tip)
        self.fields[key] = w
        self.labels[key] = label
        row = QtWidgets.QHBoxLayout()
        row.addWidget(w)
        u = QtWidgets.QLabel(unit)
        u.setStyleSheet("color:#666")
        row.addWidget(u)
        row.addStretch()
        form.addRow(label, row)
        return w

    def _txt(self, form, key, label, value):
        w = QtWidgets.QLineEdit(value)
        w.setMaximumWidth(240)
        self.fields[key] = w
        self.labels[key] = label
        form.addRow(label, w)
        return w

    def _table(self, kind, layout, reset_label="Reset table to defaults"):
        t = PropertyTable(kind)
        self.tables[kind] = t
        layout.addWidget(t)
        btn = QtWidgets.QPushButton(reset_label)
        btn.setMaximumWidth(220)
        btn.clicked.connect(lambda: self.refresh_tables(reset=True, only=kind))
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
        return t

    # ---- tabs -----------------------------------------------------------------
    def _tab_operating(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._num(f, "T_amb_low", "Low ambient (cold season)", d["T_amb_low"], "degC")
        self._num(f, "T_amb_mid", "Mid ambient", d["T_amb_mid"], "degC")
        self._num(f, "T_amb_hot", "Hot ambient", d["T_amb_hot"], "degC")
        lay.addWidget(self._group("Ambient (surroundings) temperatures", f))
        f = QtWidgets.QFormLayout()
        self._num(f, "T_water_in", "Coil inlet temperature", d["T_water_in"], "degC (highest operating)")
        self._num(f, "flow_m3h", "Coil volumetric flow", d["flow_m3h"], "m3/h")
        self._num(f, "beta_liq", "Liquid thermal expansion beta", d["beta_liq"], "1/K  (no agitator only)")
        self._num(f, "dT_natural", "Natural-convection delta-T", d["dT_natural"], "K  (no agitator only)")
        lay.addWidget(self._group("Operating conditions", f))
        f = QtWidgets.QFormLayout()
        self.cb_env = QtWidgets.QComboBox()
        for label, _h in ENV_PRESETS:
            self.cb_env.addItem(label)
        self.fields["env_index"] = self.cb_env
        f.addRow("Environment", self.cb_env)
        self._num(f, "t_ins", "Insulation thickness", d["t_ins"], "m  (0 = none)")
        self._num(f, "k_ins", "Insulation conductivity", d["k_ins"], "W/(m*K)  (0.04 mineral wool)")
        lay.addWidget(self._group("Insulation / environment", f))
        lay.addStretch()
        return w

    def _tab_agitator(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        self.cb_agitator = QtWidgets.QCheckBox("Use an agitator")
        self.cb_agitator.setChecked(d["has_agitator"])
        self.fields["has_agitator"] = self.cb_agitator
        lay.addWidget(self.cb_agitator)
        f = QtWidgets.QFormLayout()
        self.cb_ag_type = QtWidgets.QComboBox()
        for k, label in AGITATORS:
            self.cb_ag_type.addItem(f"({k}) {label}", k)
        self.fields["ag_type"] = self.cb_ag_type
        f.addRow("Impeller type", self.cb_ag_type)
        self._num(f, "N_rpm", "Rotational speed", d["N_rpm"], "RPM")
        self._num(f, "D_imp", "Impeller diameter", d["D_imp"], "m  (0 = tank radius)")
        self._num(f, "visc_corr", "Viscosity correction (mu_bulk/mu_wall)^0.14", d["visc_corr"])
        self.ag_group = self._group("Agitator settings", f)
        lay.addWidget(self.ag_group)
        lay.addWidget(QtWidgets.QLabel("Without an agitator the oil side uses natural convection "
                                       "(set beta and delta-T on the Operating tab)."))
        lay.addStretch()
        return w

    def _tab_coil_fluid(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._txt(f, "fluid_name", "Fluid name", d["fluid_name"])
        self._num(f, "T_water_low", "Lowest expected coil-fluid temperature", d["T_water_low"],
                  "degC  (follows the low ambient until you edit it)")
        lay.addWidget(self._group("Coil-side fluid (heat-transfer fluid inside the tube)", f))
        lay.addWidget(QtWidgets.QLabel("Properties at 5 points from the lowest coil-fluid temperature to the "
                                       "coil inlet temperature. Water is filled in automatically."))
        self._table("water", lay)
        lay.addStretch()
        return w

    def _tab_tank_fluid(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._txt(f, "tank_name", "Name", d["tank_name"])
        self._num(f, "T_ms", "Melt-start temperature", d["T_ms"], "degC (below = fully solid)")
        self._num(f, "T_me", "Melt-end temperature", d["T_me"], "degC (above = fully liquid)")
        self._num(f, "Hfus", "Latent heat of fusion", d["Hfus"], "J/kg")
        lay.addWidget(self._group("Tank-side fluid (working fluid / PCM)", f))
        lay.addWidget(QtWidgets.QLabel("Solid properties: low ambient to melt-start.   "
                                       "Liquid properties: melt-end to coil inlet temperature."))
        lay.addWidget(QtWidgets.QLabel("<b>Solid</b>"))
        self._table("solid", lay)
        lay.addWidget(QtWidgets.QLabel("<b>Liquid</b>"))
        self._table("liquid", lay)
        lay.addStretch()
        return w

    def _tab_hardware(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._txt(f, "wall_name", "Material name", d["wall_name"])
        self._num(f, "t_wall", "Wall thickness", d["t_wall"], "m  (0.005 = 5 mm)")
        lay.addWidget(self._group("Tank wall", f))
        self._table("wall", lay)
        f = QtWidgets.QFormLayout()
        self._txt(f, "coil_mat", "Tube material name", d["coil_mat"])
        self._num(f, "d_in", "Inner diameter", d["d_in"], "m  (0.0409 = 40.9 mm)")
        self._num(f, "d_out", "Outer diameter", d["d_out"], "m  (0.04826 = 48.3 mm)")
        self._num(f, "R_f_out", "Fouling resistance, outside", d["R_f_out"], "m2*K/W  (0.0004 typical)")
        lay.addWidget(self._group("Coil tube", f))
        self._table("tube", lay)
        lay.addStretch()
        return w

    def _tab_simulation(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._num(f, "r_inner", "Tank INNER radius", d["r_inner"], "m")
        self._num(f, "L_avail", "AXIAL length available for the coil", d["L_avail"], "m")
        lay.addWidget(self._group("Tank geometry", f))

        f = QtWidgets.QFormLayout()
        self.cb_fill = QtWidgets.QCheckBox("Fill from geometry")
        self.cb_fill.setChecked(d["fill_from_geometry"])
        self.fields["fill_from_geometry"] = self.cb_fill
        f.addRow(self.cb_fill)
        self._num(f, "M_fluid", "Fluid mass", 9000.0, "kg  (display only - see note)")
        self.lbl_fill = QtWidgets.QLabel("")
        f.addRow(self.lbl_fill)
        note = QtWidgets.QLabel("The simulation always uses tank volume x oil density. A typed mass only "
                                "appears in the report.")
        note.setWordWrap(True); note.setStyleSheet("color:#666")
        f.addRow(note)
        lay.addWidget(self._group("Tank fluid mass", f))

        f = QtWidgets.QFormLayout()
        self._num(f, "t_window_hr", "Heating window per phase", d["t_window_hr"], "h  (cold run and hot run)")
        self._num(f, "t_cooldown_hr", "Cooldown window", d["t_cooldown_hr"], "h  (off-time between runs)")
        self._num(f, "N", "Radial nodes N", d["N"], "20 - 500", integer=True,
                  tip="More nodes = finer results (each node holds total mass / N).")
        self._num(f, "dt_sec", "Timestep dt", d["dt_sec"], "s  (0.1 - 60;  1 s reference, 10 s ~ <0.1 % error)")
        self._num(f, "n_samples", "Turn samples per radius", d["n_samples"], "log-spaced; min 2", integer=True)
        lay.addWidget(self._group("Simulation", f))
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)
        lay.addStretch()
        return w

    # ---- reactive behaviour --------------------------------------------------
    def _try(self, key):
        try:
            return self.fields[key].value()
        except (ValueError, KeyError):
            return None

    def _temps_inputs(self):
        vals = {k: self._try(k) for k in ("T_amb_low", "T_water_in", "T_water_low", "T_ms", "T_me")}
        return None if any(v is None for v in vals.values()) else vals

    def _on_temps_changed(self):
        if not self._loading:
            self.refresh_tables()
            self._update_info()

    def _follow_low_ambient(self):
        if not self._tw_low_manual and not self._loading:
            self.fields["T_water_low"].setText(self.fields["T_amb_low"].text())

    def refresh_tables(self, reset=False, only=None):
        t = self._temps_inputs()
        if t is None:
            return
        temps = table_temps(t)
        name = self.fields["fluid_name"].text()
        for kind, tbl in self.tables.items():
            if only and kind != only:
                continue
            tbl.set_temps(temps[kind])
            if reset or not tbl.dirty:
                tbl.apply_defaults(temps[kind], name)

    def _agitator_toggled(self, on):
        self.ag_group.setEnabled(on)
        for k in ("beta_liq", "dT_natural"):
            self.fields[k].setEnabled(not on)

    def _fill_toggled(self, on):
        self.fields["M_fluid"].setEnabled(not on)
        self._update_info()

    def _update_info(self):
        r, L = self._try("r_inner"), self._try("L_avail")
        d_out, ns = self._try("d_out"), self._try("n_samples")
        if None in (r, L, d_out, ns) or r <= 0 or L <= 0 or d_out <= 0:
            return
        rho = self.tables["solid"].values()["rho"][0] or 920.0
        m_fill = np.pi * r ** 2 * L * rho
        self.lbl_fill.setText(f"Geometry fill: {m_fill:,.0f} kg")
        if self.cb_fill.isChecked():
            self.fields["M_fluid"].set_value(round(m_fill))
        n_max = max_turns({"L_avail": L, "d_out": d_out})
        n_pts = len(whole_turns_logspace(1, n_max, max(2, int(ns))))
        N, dt = self._try("N"), self._try("dt_sec")
        txt = (f"Coil turns: 1 to {n_max} (= axial length / tube OD).  "
               f"Sweep = 5 radii x {n_pts} turn counts x 3 ambients = {5 * n_pts * 3} cold runs, "
               f"then 3 cooldowns and 3 hot runs.")
        if N and dt and N > 0 and dt > 0:
            per_run = 3.0 * (N / 100.0) / dt
            txt += f"\nRough single-core estimate for the sweep: ~{5 * n_pts * 3 * per_run / 60:.1f} min."
        self.lbl_info.setText(txt)

    # ---- values in / out ---------------------------------------------------------
    def get_inputs(self):
        """Read the form.  Returns (inputs, errors); errors are unreadable fields."""
        inp, errors = {}, []
        for key in SCALAR_DEFAULTS:
            w = self.fields.get(key)
            if w is None:
                continue
            if isinstance(w, NumberEdit):
                try:
                    inp[key] = w.value()
                except ValueError:
                    errors.append(f"'{self.labels[key]}' is not a valid number.")
            elif isinstance(w, QtWidgets.QCheckBox):
                inp[key] = w.isChecked()
            elif key == "ag_type":
                inp[key] = int(w.currentData())
            elif isinstance(w, QtWidgets.QComboBox):
                inp[key] = int(w.currentIndex())
            else:
                inp[key] = w.text().strip()
        inp["tables"] = {k: t.values() for k, t in self.tables.items()}
        return inp, errors

    def set_inputs(self, inp):
        self._loading = True
        self._tw_low_manual = True  # a loaded / default value is explicit
        for key, w in self.fields.items():
            if key not in inp:
                continue
            v = inp[key]
            if isinstance(w, NumberEdit):
                w.set_value(v)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(v))
            elif key == "ag_type":
                w.setCurrentIndex(max(0, w.findData(int(v))))
            elif isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(int(v))
            else:
                w.setText(str(v))
        self._loading = False
        self._tw_low_manual = abs(float(inp["T_water_low"]) - float(inp["T_amb_low"])) > 1e-12
        self.refresh_tables(reset=True)
        t = self._temps_inputs()
        if t is not None:
            temps = table_temps(t)
            for kind, tbl in self.tables.items():
                loaded = (inp.get("tables") or {}).get(kind)
                if not loaded:
                    continue
                default = default_table(kind, temps[kind], inp.get("fluid_name", "water"))
                same = default is not None and all(
                    np.allclose(loaded[k], default[k], rtol=1e-9, atol=0) for k in tbl.keys)
                if not same:
                    tbl.set_values(loaded, dirty=True)
        self._agitator_toggled(self.cb_agitator.isChecked())
        self._fill_toggled(self.cb_fill.isChecked())
        self._update_info()

    def review(self):
        inp, errors = self.get_inputs()
        cfg, warnings = None, []
        if not errors:
            cfg, errors, warnings = build_cfg(inp)
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Check your inputs",
                "Please fix the following before starting:\n\n" + "\n".join("\u2022 " + e for e in errors))
            return
        dlg = ReviewDialog(summary_text(cfg, warnings), self)
        if dlg.exec():
            self.confirmed.emit(cfg, inp)

    def save_dialog(self):
        inp, errors = self.get_inputs()
        if errors:
            QtWidgets.QMessageBox.warning(self, "Cannot save", "\n".join(errors))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save inputs", "inputs.json", "JSON (*.json)")
        if path:
            save_inputs(path, inp)

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load inputs", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.set_inputs(load_inputs(path))
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Could not load", f"{path}\n\n{e}")
