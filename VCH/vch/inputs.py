"""Inputs: Simplified UI. Removes editable tables; relies on PCHIP cached data."""
import json
import numpy as np
from .geometry import max_turns
from . import materials
import traceback

ENV_PRESETS = [("Indoor  (h_air = 5 W/m2K)", 5.0),
               ("Outdoor  (h_air = 10 W/m2K)", 10.0),
               ("Outdoor windy  (h_air = 25 W/m2K)", 25.0)]

AGITATORS = [(i + 1, f"{m['name']} ({m['pattern']})") for i, m in enumerate(materials.IMPELLERS)]

SCALAR_DEFAULTS = dict(
    T_amb_low=2.0, T_amb_mid=13.0, T_amb_hot=22.0,
    T_water_in=70.0, flow_m3h=1.17, beta_liq=7e-4, dT_natural=5.0,
    has_agitator=False, ag_type=1, N_rpm=60.0, D_imp=0.0, visc_corr=1.0,
    fluid_name=list(materials.FLUIDS.keys())[0] if materials.FLUIDS else "Water", 
    tank_name=list(materials.FLUIDS.keys())[-1] if materials.FLUIDS else "RBD Palm Oil", 
    wall_name=list(materials.METALS.keys())[0] if materials.METALS else "Stainless Steel", 
    t_wall=0.005,
    coil_mat=list(materials.METALS.keys())[0] if materials.METALS else "Stainless Steel", 
    d_in=0.0409, d_out=0.04826, R_f_out=0.0004,
    env_index=0, t_ins=0.0, k_ins=0.038,
    r_inner=1.22, L_avail=2.3, fill_from_geometry=True, M_fluid=0.0,
    t_window_hr=12.0, t_cooldown_hr=12.0, N=100, dt_sec=1.0, n_samples=10,
)

def build_cfg(inp):
    errors, warnings = [], []
    g = inp
    
    if not materials.FLUIDS: return None, ["No fluids loaded. Please process materials first."], []
    if not materials.METALS: return None, ["No metals loaded. Please process materials first."], []

    if g["T_water_in"] <= g["T_amb_low"]: errors.append("Coil inlet temperature must be above low ambient.")
    if not (20 <= g["N"] <= 500): errors.append("Radial nodes N must be between 20 and 500.")
    
    # --- Pull the auto-calculated values from materials.py ---
    if g["tank_name"] not in materials.MELT_PROPS:
        return None, [f"Could not find melting data for '{g['tank_name']}'.\n\nYou MUST select a material that has a melting point (like RBD Palm Oil), not Water."], []
    
    T_ms_calc, T_me_calc, Hfus_calc = materials.MELT_PROPS[g["tank_name"]]

    if T_ms_calc >= T_me_calc: errors.append("Melt-start must be below melt-end.")
    # REMOVED: Low ambient restriction is gone
    if T_me_calc >= g["T_water_in"]: errors.append("Melt-end must be below coil inlet.")

    if errors: return None, errors, warnings

    D_imp = g["D_imp"] if g["D_imp"] > 0 else g["r_inner"]
    ag_idx = int(g["ag_type"]) - 1
    ag_C, ag_a = (materials.IMPELLERS[ag_idx]["C"], materials.IMPELLERS[ag_idx]["a"]) if 0 <= ag_idx < len(materials.IMPELLERS) else (0.54, 0.67)
    h_air = ENV_PRESETS[int(g["env_index"])][1]

    # --- CRITICAL FIX: Convert lists to numpy arrays so [:, 0] slicing works ---
    fluid_data = np.array(materials.FLUIDS.get(g["fluid_name"]))
    tank_data = np.array(materials.FLUIDS.get(g["tank_name"]))
    wall_data = np.array(materials.METALS.get(g["wall_name"]))
    coil_data = np.array(materials.METALS.get(g["coil_mat"]))
    
    if len(fluid_data) == 0: return None, [f"Coil fluid '{g['fluid_name']}' not found."], []
    if len(tank_data) == 0: return None, [f"Tank fluid '{g['tank_name']}' not found."], []
    
    solid_mask = tank_data[:, 0] <= T_ms_calc
    liquid_mask = tank_data[:, 0] >= T_me_calc
    
    if not np.any(solid_mask): return None, [f"Tank fluid '{g['tank_name']}' has no solid phase data below melt-start ({T_ms_calc}°C). Check Excel file."], []
    if not np.any(liquid_mask): return None, [f"Tank fluid '{g['tank_name']}' has no liquid phase data above melt-end ({T_me_calc}°C). Check Excel file."], []
    
    cfg = dict(
        fluid_name=g["fluid_name"],
        T_water_arr=fluid_data[:, 0], rho_water_arr=fluid_data[:, 1],
        cp_water_arr=fluid_data[:, 2], mu_water_arr=fluid_data[:, 3], k_water_arr=fluid_data[:, 4],
        
        tank_name=g["tank_name"], T_ms=T_ms_calc, T_me=T_me_calc, Hfus=Hfus_calc,
        
        T_solid_arr=tank_data[solid_mask, 0], rho_solid_arr=tank_data[solid_mask, 1],
        cp_solid_arr=tank_data[solid_mask, 2], k_solid_arr=tank_data[solid_mask, 4], mu_solid_arr=tank_data[solid_mask, 3],
        T_liquid_arr=tank_data[liquid_mask, 0], rho_liquid_arr=tank_data[liquid_mask, 1],
        cp_liquid_arr=tank_data[liquid_mask, 2], k_liquid_arr=tank_data[liquid_mask, 4], mu_liquid_arr=tank_data[liquid_mask, 3],
        
        wall_name=g["wall_name"], t_wall=g["t_wall"],
        T_wall_arr=wall_data[:, 0], rho_wall_arr=wall_data[:, 1],
        cp_wall_arr=wall_data[:, 2], k_wall_arr=wall_data[:, 3],
        
        coil_mat=g["coil_mat"], d_in=g["d_in"], d_out=g["d_out"], R_f_out=g["R_f_out"],
        T_tube_arr=coil_data[:, 0], rho_tube_arr=coil_data[:, 1],
        cp_tube_arr=coil_data[:, 2], k_tube_arr=coil_data[:, 3],
        
        h_air=h_air, t_ins=g["t_ins"], k_ins=g["k_ins"],
        r_inner=g["r_inner"], L_avail=g["L_avail"],
        T_water_in=g["T_water_in"], flow_m3h=g["flow_m3h"],
        T_amb_low=g["T_amb_low"], T_amb_mid=g["T_amb_mid"], T_amb_hot=g["T_amb_hot"],
        t_window_hr=g["t_window_hr"], t_cooldown_hr=g["t_cooldown_hr"],
        dt_sec=float(g["dt_sec"]), N=int(g["N"]), n_samples=int(g["n_samples"]),
        has_agitator=1 if g["has_agitator"] else 0,
        ag_type=int(g["ag_type"]) if g["has_agitator"] else 0,
        ag_C=float(ag_C), ag_a=float(ag_a),
        N_rpm=float(g["N_rpm"]) if g["has_agitator"] else 0.0,
        D_imp=float(D_imp) if g["has_agitator"] else 0.0,
        visc_corr=float(g["visc_corr"]) if g["has_agitator"] else 1.0,
        beta_liq=g["beta_liq"], dT_natural=g["dT_natural"],
    )

    V_tank = np.pi * cfg["r_inner"] ** 2 * cfg["L_avail"]
    m_fill = V_tank * float(np.interp(cfg["T_amb_low"], cfg["T_solid_arr"], cfg["rho_solid_arr"]))
    cfg["M_fill"] = m_fill
    cfg["M_fluid"] = m_fill if g["fill_from_geometry"] else float(g["M_fluid"])
    
    return cfg, errors, warnings

# --- PYSIDE6 UI ---
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

def _fmt(v): return str(int(v)) if isinstance(v, (int, np.integer)) else f"{float(v):.10g}"

class NumberEdit(QtWidgets.QLineEdit):
    def __init__(self, value, integer=False, width=120):
        super().__init__(_fmt(value))
        self.integer = integer
        v = QtGui.QIntValidator(self) if integer else QtGui.QDoubleValidator(self)
        v.setLocale(QtCore.QLocale.c())
        self.setValidator(v)
        self.setMaximumWidth(width)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
    def value(self): return int(self.text()) if self.integer else float(self.text())
    def set_value(self, v): self.setText(_fmt(v))

class InputsForm(QtWidgets.QWidget):
    confirmed = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fields = {}
        self.labels = {}
        d = dict(SCALAR_DEFAULTS)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._page(self._tab_operating(d)), "Operating")
        self.tabs.addTab(self._page(self._tab_agitator(d)), "Agitator")
        self.tabs.addTab(self._page(self._tab_fluids(d)), "Fluids & Materials")
        self.tabs.addTab(self._page(self._tab_hardware(d)), "Wall && coil tube")
        self.tabs.addTab(self._page(self._tab_simulation(d)), "Tank && simulation")

        bar = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load inputs...")
        b_save = QtWidgets.QPushButton("Save inputs...")
        b_load.clicked.connect(self.load_dialog)
        b_save.clicked.connect(self.save_dialog)
        
        self.btn_go = QtWidgets.QPushButton("Start Phase A  \u25B6")
        self.btn_go.setObjectName("primary")
        self.btn_go.setMinimumHeight(34)
        self.btn_go.clicked.connect(self.review)
        
        bar.addWidget(b_load); bar.addWidget(b_save); bar.addStretch(); bar.addWidget(self.btn_go)
        
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self.tabs)
        root.addLayout(bar)

        self._agitator_toggled(self.cb_agitator.isChecked())
        self._fill_toggled(self.cb_fill.isChecked())

    @staticmethod
    def _page(widget):
        sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sc.setWidget(widget); return sc

    @staticmethod
    def _group(title, layout):
        g = QtWidgets.QGroupBox(title); g.setLayout(layout); return g

    def _num(self, form, key, label, value, unit="", integer=False):
        w = NumberEdit(value, integer)
        self.fields[key] = w; self.labels[key] = label
        row = QtWidgets.QHBoxLayout(); row.addWidget(w)
        u = QtWidgets.QLabel(unit); u.setStyleSheet("color:#666"); row.addWidget(u); row.addStretch()
        form.addRow(label, row); return w

    def _combo(self, form, key, label, items, current):
        w = QtWidgets.QComboBox(); w.addItems(items)
        if current in items: w.setCurrentText(current)
        w.setMaximumWidth(280)
        self.fields[key] = w; self.labels[key] = label
        form.addRow(label, w); return w

    def _tab_operating(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._num(f, "T_amb_low", "Low ambient", d["T_amb_low"], "degC")
        self._num(f, "T_amb_mid", "Mid ambient", d["T_amb_mid"], "degC")
        self._num(f, "T_amb_hot", "Hot ambient", d["T_amb_hot"], "degC")
        lay.addWidget(self._group("Ambient Temperatures", f))
        
        f = QtWidgets.QFormLayout()
        self._num(f, "T_water_in", "Coil inlet temperature", d["T_water_in"], "degC")
        self._num(f, "flow_m3h", "Coil volumetric flow", d["flow_m3h"], "m3/h")
        # ADDED BACK: These are required for the natural convection physics equations
        self._num(f, "beta_liq", "Thermal expansion coeff (beta)", d["beta_liq"], "1/K")
        self._num(f, "dT_natural", "Driving delta-T", d["dT_natural"], "K")
        lay.addWidget(self._group("Operating Conditions", f))
        
        f = QtWidgets.QFormLayout()
        self.cb_env = QtWidgets.QComboBox()
        for label, _h in ENV_PRESETS: self.cb_env.addItem(label)
        self.fields["env_index"] = self.cb_env
        f.addRow("Environment", self.cb_env)
        self._num(f, "t_ins", "Insulation thickness", d["t_ins"], "m")
        self._num(f, "k_ins", "Insulation conductivity", d["k_ins"], "W/(m*K)")
        lay.addWidget(self._group("Insulation / Environment", f))
        lay.addStretch(); return w

    def _tab_agitator(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        self.cb_agitator = QtWidgets.QCheckBox("Use an agitator")
        self.cb_agitator.setChecked(d["has_agitator"])
        self.fields["has_agitator"] = self.cb_agitator
        lay.addWidget(self.cb_agitator)
        f = QtWidgets.QFormLayout()
        self.cb_ag_type = QtWidgets.QComboBox()
        if not AGITATORS:
            self.cb_ag_type.addItem("Default", 1)
        else:
            for k, label in AGITATORS: self.cb_ag_type.addItem(label, k)
        self.fields["ag_type"] = self.cb_ag_type
        f.addRow("Impeller type", self.cb_ag_type)
        self._num(f, "N_rpm", "Rotational speed", d["N_rpm"], "RPM")
        self._num(f, "D_imp", "Impeller diameter", d["D_imp"], "m")
        self._num(f, "visc_corr", "Viscosity correction", d["visc_corr"])
        self.ag_group = self._group("Agitator settings", f)
        lay.addWidget(self.ag_group); lay.addStretch(); return w

    def _tab_fluids(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        
        f = QtWidgets.QFormLayout()
        self._combo(f, "fluid_name", "Coil Fluid", list(materials.FLUIDS.keys()), d["fluid_name"])
        lay.addWidget(self._group("Coil Fluid (Heating Side)", f))
        
        f = QtWidgets.QFormLayout()
        self._combo(f, "tank_name", "Tank Fluid", list(materials.FLUIDS.keys()), d["tank_name"])
        lay.addWidget(self._group("Tank Fluid (Phase Change Side)", f))
        
        note = QtWidgets.QLabel("Material properties are automatically loaded from the VCH Framework.xlsx database "
                                "and smoothed using PCHIP curve fitting. Melt temperatures and latent heat are extracted automatically.\n"
                                "No manual property entry is required.")
        note.setWordWrap(True); note.setStyleSheet("color:#666; margin-top: 15px; font-style: italic;")
        lay.addWidget(note)
        lay.addStretch(); return w

    def _tab_hardware(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._combo(f, "wall_name", "Tank wall material", list(materials.METALS.keys()), d["wall_name"])
        self._num(f, "t_wall", "Wall thickness", d["t_wall"], "m")
        lay.addWidget(self._group("Tank Wall", f))
        
        f = QtWidgets.QFormLayout()
        self._combo(f, "coil_mat", "Coil tube material", list(materials.METALS.keys()), d["coil_mat"])
        self._num(f, "d_in", "Inner diameter", d["d_in"], "m")
        self._num(f, "d_out", "Outer diameter", d["d_out"], "m")
        self._num(f, "R_f_out", "Fouling resistance", d["R_f_out"], "m2*K/W")
        lay.addWidget(self._group("Coil Tube", f))
        lay.addStretch(); return w

    def _tab_simulation(self, d):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QFormLayout()
        self._num(f, "r_inner", "Tank inner radius", d["r_inner"], "m")
        self._num(f, "L_avail", "Coil axial length", d["L_avail"], "m")
        lay.addWidget(self._group("Tank Geometry", f))
        f = QtWidgets.QFormLayout()
        self.cb_fill = QtWidgets.QCheckBox("Fill mass from geometry")
        self.cb_fill.setChecked(d["fill_from_geometry"]); self.fields["fill_from_geometry"] = self.cb_fill
        f.addRow(self.cb_fill)
        self._num(f, "M_fluid", "Fluid mass", 9000.0, "kg")
        lay.addWidget(self._group("Tank Fluid Mass", f))
        f = QtWidgets.QFormLayout()
        self._num(f, "t_window_hr", "Heating window", d["t_window_hr"], "h")
        self._num(f, "t_cooldown_hr", "Cooldown window", d["t_cooldown_hr"], "h")
        self._num(f, "N", "Radial nodes N", d["N"], "int", True)
        self._num(f, "dt_sec", "Timestep dt", d["dt_sec"], "s")
        self._num(f, "n_samples", "Turn samples", d["n_samples"], "int", True)
        lay.addWidget(self._group("Simulation", f))
        lay.addStretch(); return w

    def _agitator_toggled(self, on):
        if hasattr(self, 'ag_group'): self.ag_group.setEnabled(on)
    def _fill_toggled(self, on):
        if "M_fluid" in self.fields: self.fields["M_fluid"].setEnabled(not on)

    def get_inputs(self):
        inp, errors = {}, []
        for key in SCALAR_DEFAULTS:
            w = self.fields.get(key)
            if w is None: continue
            if isinstance(w, NumberEdit):
                try: inp[key] = w.value()
                except ValueError: errors.append(f"'{self.labels[key]}' is invalid.")
            elif isinstance(w, QtWidgets.QCheckBox): inp[key] = w.isChecked()
            elif key == "ag_type": 
                data = w.currentData()
                inp[key] = int(data) if data is not None else 1
            elif isinstance(w, QtWidgets.QComboBox): inp[key] = int(w.currentIndex()) if key == "env_index" else w.currentText()
            else: inp[key] = w.text().strip()
        return inp, errors

    def set_inputs(self, inp):
        for key, w in self.fields.items():
            if key not in inp: continue
            v = inp[key]
            if isinstance(w, NumberEdit): w.set_value(v)
            elif isinstance(w, QtWidgets.QCheckBox): w.setChecked(bool(v))
            elif key == "ag_type": w.setCurrentIndex(max(0, w.findData(int(v))))
            elif isinstance(w, QtWidgets.QComboBox):
                if key == "env_index": w.setCurrentIndex(int(v))
                else: w.setCurrentText(str(v))
            else: w.setText(str(v))

    def review(self):
        try:
            inp, errors = self.get_inputs()
            cfg, warnings = None, []
            if not errors: 
                cfg, errors, warnings = build_cfg(inp)
            
            if errors:
                err_msg = "\n\n".join(errors)
                print("INPUT ERRORS DETECTED:\n", err_msg)
                QtWidgets.QMessageBox.critical(self, "Cannot Start Simulation", err_msg)
                return
                
            if cfg is None:
                QtWidgets.QMessageBox.critical(self, "Cannot Start Simulation", "Unknown error building configuration.")
                return

            self.confirmed.emit(cfg, inp)
        except Exception as e:
            err_trace = traceback.format_exc()
            print("PYTHON CRASH:\n", err_trace)
            QtWidgets.QMessageBox.critical(self, "Python Crash", f"An error occurred before starting:\n\n{str(e)}\n\nSee terminal for details.")
    
    def save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", "inputs.json", "JSON (*.json)")
        if path:
            with open(path, "w") as f: json.dump(self.get_inputs()[0], f, indent=2)
            
    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if path:
            with open(path, "r") as f: self.set_inputs(json.load(f))