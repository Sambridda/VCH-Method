import io
import multiprocessing
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, QThread

from vch import inputs, post, materials
from vch.physics import (
    build_material_tables, run_cold_sweep, run_2d_simulation, 
    default_workers, Cancelled
)
from vch.physics.engine_2d import warm_up_2d
from vch.geometry import build_geometry
from vch.resource_monitor import WorkerMonitor

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
OUT_ROOT = APP_DIR / "out"

STYLE = """
QPushButton#primary { background:#1a4d7a; color:white; border:none; border-radius:4px; padding:6px 16px; font-weight:600; }
QPushButton#primary:hover { background:#256aa3; }
QPushButton#primary:disabled { background:#9fb3c6; }
QGroupBox { font-weight:600; margin-top:10px; }
QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }
QProgressBar { border:1px solid #bbb; border-radius:4px; text-align:center; height:22px; }
QProgressBar::chunk { background:#2f7fc1; }
"""

# ============================================================
# BACKGROUND WORKERS
# ============================================================
class MaterialWorker(QThread):
    done = Signal(bool, str)
    def run(self):
        try:
            materials.process_materials(force=True)
            self.done.emit(True, "")
        except Exception:
            self.done.emit(False, traceback.format_exc())

class PhaseAWorker(QThread):
    progress = Signal(int, int, str)
    workers_snapshot = Signal(object)
    done = Signal(object)
    failed = Signal(str)
    def __init__(self, cfg, outdir): 
        super().__init__()
        self.cfg, self.outdir = cfg, outdir
    def run(self):
        try:
            cfg = self.cfg
            mat = build_material_tables(cfg)
            cfg["_mat"] = mat
            geom, n_max, radii, turns = build_geometry(cfg)
            presets = [("low", cfg["T_amb_low"]), ("mid", cfg["T_amb_mid"]), ("hot", cfg["T_amb_hot"])]
            mon = WorkerMonitor(interval=1.0)
            mon.on_snapshot = self.workers_snapshot.emit
            
            cold_rows = run_cold_sweep(cfg, mat, geom, presets,
                                       progress=lambda d, t, m: self.progress.emit(d, t, m), 
                                       monitor=mon)
                                       
            menu = post.write_phase_a(self.outdir, cfg, cold_rows, geom, radii, presets)
            self.done.emit(dict(mat=mat, geom=geom, radii=radii, presets=presets, cold_rows=cold_rows, menu=menu))
        except Exception: 
            self.failed.emit(traceback.format_exc())

class PhaseCDWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)
    def __init__(self, cfg, study, choice, outdir): 
        super().__init__()
        self.cfg, self.study, self.choice, self.outdir = cfg, study, choice, outdir

    def run(self):
        try:
            R, n = self.choice["R"], self.choice["n"]
            
            res = run_2d_simulation(
                self.cfg, self.study["mat"], R, n, None, 
                drain_diameter=self.cfg["drain_diameter"],
                drain_location=(int(self.cfg["drain_r_idx"]), int(self.cfg["drain_z_idx"])),
                outdir=self.outdir,
                progress_callback=lambda d, t, m: self.progress.emit(d, t, m)
            )
            
            html_path = post.write_report_html(
                self.outdir, self.cfg, R, n, 
                cold_at_selected={}, cool_reports={}, hot_rows=[], presets=[], 
                results_2d=res
            )
            
            res["text"] = f"2D Draining Simulation Complete.\n\nTotal Time: {res['total_hr']:.1f} h\nTotal Drained: {res['total_extracted_kg']:.1f} kg\nFinal Mass: {res['final_mass']:.1f} kg\n\nSee the HTML report and the graph tabs for full details."
            res["index_html"] = html_path
            res["R"] = R
            res["n"] = n
            res["L"] = self.choice["L"]
            
            self.done.emit(res)
        except Exception: 
            self.failed.emit(traceback.format_exc())

# ============================================================
# UI PAGES
# ============================================================
class ScaledImage(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        self._pm = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.setMinimumSize(160, 120)
    def set_image(self, path):
        self._pm = QtGui.QPixmap(str(path))
        self._rescale()
    def _rescale(self):
        if self._pm is not None and not self._pm.isNull():
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)

def image_tabs(names, folder):
    tabs = QtWidgets.QTabWidget()
    for name in names:
        img = ScaledImage()
        p = Path(folder) / name
        if p.exists(): img.set_image(p)
        tabs.addTab(img, name.replace(".png", ""))
    return tabs

def open_path(path):
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

class LandingPage(QtWidgets.QWidget):
    go_to_sim = Signal()
    show_graphs = Signal()
    
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QtWidgets.QLabel("VCH Framework")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #1a4d7a; margin-bottom: 20px;")
        lay.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btn_mat = QtWidgets.QPushButton("1. Process Materials Database")
        btn_mat.setMinimumSize(300, 50)
        btn_mat.clicked.connect(self.run_materials)
        
        btn_sim = QtWidgets.QPushButton("2. Run Simulation")
        btn_sim.setMinimumSize(300, 50)
        btn_sim.setObjectName("primary")
        btn_sim.clicked.connect(self.go_to_sim.emit)
        
        lay.addWidget(btn_mat, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(btn_sim, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: #666; margin-top: 15px;")
        lay.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignCenter)

    def run_materials(self):
        self.status.setText("Processing Excel and generating PCHIP curves...")
        QtWidgets.QApplication.processEvents()
        worker = MaterialWorker()
        worker.done.connect(self._mat_done)
        worker.start()
        self._worker = worker

    def _mat_done(self, success, error_msg):
        if success:
            self.status.setText("Materials updated! Opening graphs...")
            self.show_graphs.emit()
        else:
            self.status.setText("Error processing materials.")
            QtWidgets.QMessageBox.critical(self, "Processing Error", error_msg)

class MaterialGraphsPage(QtWidgets.QWidget):
    back_to_menu = Signal()
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Material Property Curves")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1a4d7a;")
        btn_back = QtWidgets.QPushButton("\u25C0 Back to Menu")
        btn_back.clicked.connect(self.back_to_menu.emit)
        head.addWidget(title); head.addStretch(); head.addWidget(btn_back)
        lay.addLayout(head)
        
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        self.container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.container)
        scroll.setWidget(self.container)
        lay.addWidget(scroll)

    def load_graphs(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w: w.deleteLater()
        if not os.path.exists(materials.MAT_DIR): return
        images = sorted([f for f in os.listdir(materials.MAT_DIR) if f.endswith('.png')])
        row, col = 0, 0
        for img_name in images:
            lbl = QtWidgets.QLabel()
            pix = QtGui.QPixmap(os.path.join(materials.MAT_DIR, img_name))
            lbl.setPixmap(pix.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            frame = QtWidgets.QGroupBox(img_name.replace('.png', '').replace('_', ' ').title())
            flay = QtWidgets.QVBoxLayout(frame); flay.addWidget(lbl)
            self.grid.addWidget(frame, row, col)
            col += 1
            if col > 1: col = 0; row += 1

class ProgressPage(QtWidgets.QWidget):
    cancel_requested = Signal()

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel("")
        f = self.title.font(); f.setPointSize(f.pointSize() + 3); f.setBold(True)
        self.title.setFont(f)
        self.status = QtWidgets.QLabel("")
        self.bar = QtWidgets.QProgressBar()
        self.eta = QtWidgets.QLabel("")
        self.cores_label = QtWidgets.QLabel("")
        
        self.worker_table = QtWidgets.QTableWidget(0, 7)
        self.worker_table.setHorizontalHeaderLabels(["#", "PID", "State", "Task", "Done", "CPU s", "Utilization"])
        self.worker_table.verticalHeader().setVisible(False)
        self.worker_table.verticalHeader().setDefaultSectionSize(26)
        self.worker_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.worker_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.worker_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.worker_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.worker_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_requested)
        
        for w in (self.title, self.status, self.bar, self.eta, self.cores_label, self.worker_table, self.log):
            lay.addWidget(w)
        lay.addWidget(self.btn_cancel, alignment=Qt.AlignmentFlag.AlignRight)
        self._t0 = time.time()

    def start(self, title, first_message, show_workers=False):
        self.title.setText(title)
        self.status.setText(first_message)
        self.bar.setRange(0, 100); self.bar.setValue(0)
        self.eta.setText("")
        self.log.clear()
        self.btn_cancel.setEnabled(True)
        self.worker_table.setVisible(show_workers)
        self.cores_label.setVisible(show_workers)
        self.worker_table.setRowCount(0)
        self.cores_label.setText("")
        self._t0 = time.time()
        self.log_line(first_message)

    def log_line(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def update_progress(self, done, total, msg):
        total = max(1, total)
        self.bar.setRange(0, total); self.bar.setValue(done)
        elapsed = time.time() - self._t0
        eta = "--"
        if done > 0 and elapsed > 0 and done < total:
            s = elapsed * (total - done) / done
            eta = f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"
        self.eta.setText(f"{done} / {total}    elapsed {elapsed:.0f} s    ETA {eta}")
        if msg:
            self.status.setText(msg)
            self.log_line(msg)

    def update_workers(self, snap):
        t = self.worker_table
        if t.rowCount() != len(snap):
            t.setRowCount(len(snap))
            header_h = t.horizontalHeader().height()
            row_h = t.verticalHeader().defaultSectionSize()
            t.setFixedHeight(header_h + (len(snap) * row_h) + 4)
            
        for r, w in enumerate(snap):
            vals = [str(w["slot"]), str(w["pid"]), w["state"], str(w["task"]) if w["task"] is not None else "-",
                    str(w["done"]), f"{w['cpu_busy']:.1f}", f"{100*w['utilization']:.0f}%"]
            for c, v in enumerate(vals):
                it = t.item(r, c)
                if it is None: it = QtWidgets.QTableWidgetItem(); t.setItem(r, c, it)
                it.setText(v); it.setTextAlignment(Qt.AlignmentFlag.AlignCenter); it.setForeground(QtGui.QColor(20, 20, 20))
            color = {"running": QtGui.QColor(214, 238, 214), "idle": QtGui.QColor(255, 244, 200), "starting": QtGui.QColor(230, 230, 230)}.get(w["state"])
            if color:
                for c in range(t.columnCount()): t.item(r, c).setBackground(color)
                    
        busy = sum(w["utilization"] for w in snap)
        self.cores_label.setText(f"Cores busy: {busy:.2f} / {os.cpu_count()} logical   —   {len(snap)} workers")

# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QtWidgets.QMainWindow):
    TAB_INPUTS, TAB_A, TAB_PICK, TAB_RESULTS = range(4)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VCH Framework")
        self.resize(1180, 820)
        self.cfg = None
        self.study = None
        self.outdir = None
        self.worker = None
        self.menu = []
        self.form = None

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.landing = LandingPage()
        self.landing.go_to_sim.connect(self.show_inputs)
        self.landing.show_graphs.connect(self.show_material_graphs)
        self.stack.addWidget(self.landing)

        self.mat_graphs = MaterialGraphsPage()
        self.mat_graphs.back_to_menu.connect(lambda: self.stack.setCurrentWidget(self.landing))
        self.stack.addWidget(self.mat_graphs)

        self.page_a = ProgressPage()
        self.page_a.cancel_requested.connect(self.cancel_worker)
        self.stack.addWidget(self.page_a)

        self.picker = self._build_pick_page()
        self.stack.addWidget(self.picker)

        self.results_page = QtWidgets.QStackedWidget()
        self.page_cd = ProgressPage()
        self.page_cd.cancel_requested.connect(self.cancel_worker)
        self.results_box = QtWidgets.QWidget()
        self.results_lay = QtWidgets.QVBoxLayout(self.results_box)
        self.results_page.addWidget(self.page_cd)
        self.results_page.addWidget(self.results_box)
        self.stack.addWidget(self.results_page)

        self.rebuild_form()
        self.statusBar().showMessage("Ready.")
        self.showMaximized()

    def show_material_graphs(self):
        self.mat_graphs.load_graphs()
        self.stack.setCurrentWidget(self.mat_graphs)
        self.rebuild_form()

    def rebuild_form(self):
        if self.form is not None:
            self.stack.removeWidget(self.form)
            self.form.deleteLater()
        self.form = inputs.InputsForm()
        self.form.confirmed.connect(self.start_phase_a)
        self.stack.addWidget(self.form)

    def show_inputs(self):
        if not materials.FLUIDS:
            QtWidgets.QMessageBox.warning(self, "Materials Missing", "No fluids loaded. Please process the materials database first.")
            return
        self.stack.setCurrentWidget(self.form)
        self.statusBar().showMessage("Enter simulation parameters.")

    def start_phase_a(self, cfg, inp):
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = str(OUT_ROOT / f"run_{ts}")
        os.makedirs(self.outdir, exist_ok=True)

        self.stack.setCurrentWidget(self.page_a)
        self.page_a.start("Phase A - Cold Sweep", f"Output folder: {self.outdir}", show_workers=True)
        
        self.page_a.log_line(f"Using up to {default_workers()} worker process(es) ({os.cpu_count()} CPU cores detected).")
        self.statusBar().showMessage("Phase A running...")

        self.worker = PhaseAWorker(cfg, self.outdir)
        self.worker.progress.connect(self.page_a.update_progress)
        self.worker.workers_snapshot.connect(self.page_a.update_workers)
        self.worker.done.connect(self._phase_a_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _phase_a_done(self, study):
        self.study = study
        self.menu = study["menu"]
        self._fill_picker()
        self.stack.setCurrentWidget(self.picker)
        self.statusBar().showMessage("Phase A complete. Please select a coil.")

    def _build_pick_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.addWidget(QtWidgets.QLabel("<b>Phase B - Pick the coil to carry forward:</b>"))
        self.pick_table = QtWidgets.QTableWidget(0, 7)
        self.pick_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.pick_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.pick_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pick_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pick_table.itemSelectionChanged.connect(self._pick_changed)
        self.pick_table.cellDoubleClicked.connect(lambda *_: self.start_phase_cd())

        self.pick_images = QtWidgets.QTabWidget()
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.pick_table); split.addWidget(self.pick_images)
        lay.addWidget(split, 1)
        
        bar = QtWidgets.QHBoxLayout()
        self.lbl_pick = QtWidgets.QLabel("Select a row.")
        self.btn_cd = QtWidgets.QPushButton("Run 2D Draining Simulation  \u25B6"); self.btn_cd.setObjectName("primary")
        self.btn_cd.setEnabled(False)
        self.btn_cd.clicked.connect(self.start_phase_cd)
        bar.addWidget(self.lbl_pick, 1); bar.addWidget(self.btn_cd)
        lay.addLayout(bar)
        return page

    def _fill_picker(self):
        t = self.pick_table
        t.clearContents()
        t.setRowCount(len(self.menu))
        t.setHorizontalHeaderLabels(["idx", "R (m)", "n", "L (m)", "liquid kg low", "liquid kg mid", "liquid kg hot"])
        best = {k: max(m[k] for m in self.menu) for k in ("liq_low", "liq_mid", "liq_hot")}
        for i, m in enumerate(self.menu):
            vals = [f"{m['idx']}", f"{m['R']:.3f}", f"{m['n']}", f"{m['L']:.1f}",
                    f"{m['liq_low']:.0f}", f"{m['liq_mid']:.0f}", f"{m['liq_hot']:.0f}"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                key = ("liq_low", "liq_mid", "liq_hot")[c - 4] if c >= 4 else None
                if key and m[key] >= best[key] - 1e-9 and best[key] > 0:
                    it.setBackground(QtGui.QColor(214, 238, 214)); it.setForeground(QtGui.QColor(20, 20, 20))
                t.setItem(i, c, it)
        t.clearSelection()
        self.btn_cd.setEnabled(False)
        self.lbl_pick.setText("Select a row.")
        self.pick_images.clear()
        for name in ("cold_low.png", "cold_mid.png", "cold_hot.png"):
            img = ScaledImage(); img.set_image(Path(self.outdir) / name)
            self.pick_images.addTab(img, name.replace(".png", "").replace("cold_", "cold start: "))

    def _pick_changed(self):
        rows = self.pick_table.selectionModel().selectedRows()
        if not rows:
            self.btn_cd.setEnabled(False)
            return
        m = self.menu[rows[0].row()]
        self.lbl_pick.setText(f"Selected: R = {m['R']:.3f} m   n = {m['n']}   L = {m['L']:.1f} m")
        self.btn_cd.setEnabled(True)

    def start_phase_cd(self):
        rows = self.pick_table.selectionModel().selectedRows()
        if not rows: return
        choice = self.menu[rows[0].row()]
        
        rho_start = float(np.interp(self.cfg["T_start"], self.study["mat"]["T_grid"], self.study["mat"]["rho_grid"]))
        V_tank_full = np.pi * self.cfg["r_inner"]**2 * self.cfg["tank_height"]
        V_target = self.cfg["M_fluid"] / max(1e-6, rho_start)
        self.cfg["M_fluid_effective"] = self.cfg["M_fluid"] if V_target <= V_tank_full else V_tank_full * rho_start

        self.results_page.setCurrentWidget(self.page_cd)
        self.stack.setCurrentWidget(self.results_page)
        self.page_cd.start("Phase C & D - 2D Draining", f"Coil R={choice['R']:.2f}, n={choice['n']}", show_workers=False)
        self.statusBar().showMessage("2D Simulation running...")

        self.worker = PhaseCDWorker(self.cfg, self.study, choice, self.outdir)
        self.worker.progress.connect(self.page_cd.update_progress)
        self.worker.done.connect(self._phase_cd_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _phase_cd_done(self, res):
        self._build_results(res)
        self.results_page.setCurrentWidget(self.results_box)
        self.statusBar().showMessage(f"Done - 2D results saved to {self.outdir}")

    def _build_results(self, res):
        while self.results_lay.count():
            w = self.results_lay.takeAt(0).widget()
            if w: w.deleteLater()
            
        head = QtWidgets.QLabel(f"<b>Finished.</b> Selected coil: R = {res['R']:.3f} m, n = {res['n']}, L = {res['L']:.1f} m<br>Files in <code>{self.outdir}</code>")
        head.setTextFormat(Qt.TextFormat.RichText); head.setWordWrap(True)
        self.results_lay.addWidget(head)

        text = QtWidgets.QPlainTextEdit(res["text"])
        text.setReadOnly(True)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        
        tabs = QtWidgets.QTabWidget()
        if "curve_path" in res: tabs.addTab(self._img(res["curve_path"]), "Drain Curve")
        
        for idx, m in enumerate(res.get("cycle_metrics", [])):
            c_tabs = QtWidgets.QTabWidget()
            if idx < len(res.get("heatmap_paths", [])): c_tabs.addTab(self._img(res["heatmap_paths"][idx]), "Lateral Temp")
            if idx < len(res.get("massmap_paths", [])): c_tabs.addTab(self._img(res["massmap_paths"][idx]), "Lateral Mass")
            if idx < len(res.get("topmass_paths", [])): c_tabs.addTab(self._img(res["topmass_paths"][idx]), "Top-Down Mass")
            if idx < len(res.get("toptemp_paths", [])): c_tabs.addTab(self._img(res["toptemp_paths"][idx]), "Top-Down Temp")
            if idx < len(res.get("phasebar_paths", [])): c_tabs.addTab(self._img(res["phasebar_paths"][idx]), "Phase Dist")
            tabs.addTab(c_tabs, f"Cycle {m['cycle']}")

        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(text)
        split.addWidget(tabs)
        text.setMinimumWidth(430); split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setSizes([400, 740])
        self.results_lay.addWidget(split, 1)

        bar = QtWidgets.QHBoxLayout()
        b_html = QtWidgets.QPushButton("Open report (index.html)")
        b_html.setObjectName("primary")
        b_html.clicked.connect(lambda: open_path(res["index_html"]))
        b_dir = QtWidgets.QPushButton("Open output folder")
        b_dir.clicked.connect(lambda: open_path(self.outdir))
        b_new = QtWidgets.QPushButton("New study")
        b_new.clicked.connect(lambda: self.stack.setCurrentWidget(self.landing))
        for b in (b_html, b_dir): bar.addWidget(b)
        bar.addStretch(); bar.addWidget(b_new)
        self.results_lay.addLayout(bar)

    def _img(self, p):
        l = ScaledImage(); l.set_image(p)
        return l

    def _failed(self, tb):
        QtWidgets.QMessageBox.critical(self, "Error", tb)
        self.stack.setCurrentWidget(self.landing)

    def cancel_worker(self):
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.statusBar().showMessage("Cancelling...")

def main():
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion"); app.setStyleSheet(STYLE)
    
    # COMPILE THE 2D KERNEL HERE BEFORE THE UI SHOWS
    warm_up_2d() 
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()