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

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, QThread

from vch import inputs, post, materials
from vch.physics import build_material_tables
from vch.geometry import build_geometry
from vch.sweep import run_cold_sweep, run_hot, run_heat, default_workers, Cancelled
from vch.cooldown import run_cooldowns, run_cool
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

class MaterialWorker(QThread):
    done = Signal(bool, str)
    def run(self):
        try:
            materials.process_materials(force=True)
            self.done.emit(True, "")
        except Exception:
            self.done.emit(False, traceback.format_exc())

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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VCH Framework")
        self.resize(1180, 820)
        self.cfg = None; self.study = None; self.outdir = None; self.worker = None; self.menu = []; self.form = None

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.landing = LandingPage()
        self.landing.go_to_sim.connect(self.show_inputs)
        self.landing.show_graphs.connect(self.show_material_graphs)
        self.stack.addWidget(self.landing)

        self.mat_graphs = MaterialGraphsPage()
        self.mat_graphs.back_to_menu.connect(lambda: self.stack.setCurrentWidget(self.landing))
        self.stack.addWidget(self.mat_graphs)

        self.page_a = self._build_progress_page("Phase A: cold sweep")
        self.stack.addWidget(self.page_a)

        self.picker = self._build_pick_page()
        self.stack.addWidget(self.picker)

        self.results_page = QtWidgets.QStackedWidget()
        self.page_cd = self._build_progress_page("Phases C & D")
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
        # We also need to rebuild the form so the new dropdowns pick up the freshly generated materials
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

    def _build_progress_page(self, title):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lbl = QtWidgets.QLabel(title); lbl.setStyleSheet("font-size:18px; font-weight:bold;")
        bar = QtWidgets.QProgressBar()
        eta = QtWidgets.QLabel("")
        cores = QtWidgets.QLabel("")
        table = QtWidgets.QTableWidget(0, 7)
        table.setHorizontalHeaderLabels(["#", "PID", "State", "Task", "Done", "CPU s", "Utilization"])
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        log = QtWidgets.QPlainTextEdit(); log.setReadOnly(True)
        btn = QtWidgets.QPushButton("Cancel")
        
        for w in (lbl, bar, eta, cores, table, log): lay.addWidget(w)
        lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        page.lbl, page.bar, page.eta, page.cores = lbl, bar, eta, cores
        page.table, page.log, page.btn = table, log, btn
        return page

    def start_phase_a(self, cfg, inp):
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = str(OUT_ROOT / f"run_{ts}")
        os.makedirs(self.outdir, exist_ok=True)

        self.stack.setCurrentWidget(self.page_a)
        self.page_a.lbl.setText(f"Phase A - Cold Sweep (Output: {self.outdir})")
        self.statusBar().showMessage("Phase A running...")

        self.worker = PhaseAWorker(cfg, self.outdir)
        self.worker.progress.connect(lambda d, t, m: self._update_prog(self.page_a, d, t, m))
        self.worker.workers_snapshot.connect(lambda s: self._update_workers(self.page_a, s))
        self.worker.done.connect(self._phase_a_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _update_prog(self, page, done, total, msg):
        page.bar.setRange(0, total); page.bar.setValue(done)
        page.eta.setText(f"{done} / {total}  -  {msg}")
        if msg: page.log.appendPlainText(msg)

    def _update_workers(self, page, snap):
        t = page.table
        if t.rowCount() != len(snap): t.setRowCount(len(snap))
        busy = 0
        for r, w in enumerate(snap):
            busy += w['utilization']
            vals = [str(w["slot"]), str(w["pid"]), w["state"], str(w["task"]), str(w["done"]), f"{w['cpu_busy']:.1f}", f"{100*w['utilization']:.0f}%"]
            for c, v in enumerate(vals):
                it = t.item(r, c)
                if it is None: it = QtWidgets.QTableWidgetItem(); t.setItem(r, c, it)
                it.setText(v)
        page.cores.setText(f"Cores Busy: {busy:.2f} / {os.cpu_count()}")

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
        self.pick_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        
        self.pick_images = QtWidgets.QTabWidget()
        split = QtWidgets.QSplitter()
        split.addWidget(self.pick_table); split.addWidget(self.pick_images)
        lay.addWidget(split, 1)
        
        btn = QtWidgets.QPushButton("Run Phases C & D  \u25B6"); btn.setObjectName("primary")
        btn.clicked.connect(self.start_phase_cd)
        lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)
        return page

    def _fill_picker(self):
        t = self.pick_table
        t.setRowCount(len(self.menu))
        best = {k: max(m[k] for m in self.menu) for k in ("liq_low", "liq_mid", "liq_hot")}
        for i, m in enumerate(self.menu):
            vals = [f"{m['idx']}", f"{m['R']:.3f}", f"{m['n']}", f"{m['L']:.1f}", f"{m['liq_low']:.0f}", f"{m['liq_mid']:.0f}", f"{m['liq_hot']:.0f}"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c >= 4 and m[("liq_low","liq_mid","liq_hot")[c-4]] == best[("liq_low","liq_mid","liq_hot")[c-4]]:
                    it.setBackground(QtGui.QColor(214, 238, 214))
                t.setItem(i, c, it)
        for name in ("cold_low.png", "cold_mid.png", "cold_hot.png"):
            self.pick_images.addTab(self._img(Path(self.outdir)/name), name)

    def _img(self, p):
        l = QtWidgets.QLabel(); l.setPixmap(QtGui.QPixmap(str(p)).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio))
        return l

    def start_phase_cd(self):
        rows = self.pick_table.selectionModel().selectedRows()
        if not rows: return
        choice = self.menu[rows[0].row()]
        
        self.results_page.setCurrentWidget(self.page_cd)
        self.stack.setCurrentWidget(self.results_page)
        self.page_cd.lbl.setText(f"Phases C & D (R={choice['R']:.2f}, n={choice['n']})")
        
        self.worker = PhaseCDWorker(self.cfg, self.study, choice, self.outdir)
        self.worker.progress.connect(lambda d, t, m: self._update_prog(self.page_cd, d, t, m))
        self.worker.done.connect(self._phase_cd_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _phase_cd_done(self, res):
        self._build_results(res)
        self.results_page.setCurrentWidget(self.results_box)

    def _build_results(self, res):
        while self.results_lay.count():
            w = self.results_lay.takeAt(0).widget()
            if w: w.deleteLater()
        self.results_lay.addWidget(QtWidgets.QLabel(f"<b>Finished.</b> Outputs in: <code>{self.outdir}</code>"))
        text = QtWidgets.QPlainTextEdit(res["text"]); text.setReadOnly(True)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        
        split = QtWidgets.QSplitter()
        split.addWidget(text)
        it = QtWidgets.QTabWidget()
        for p in post.PNG_NAMES: it.addTab(self._img(Path(self.outdir)/p), p)
        split.addWidget(it)
        self.results_lay.addWidget(split, 1)
        
        btn = QtWidgets.QPushButton("Open index.html")
        btn.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(res["index_html"])))
        self.results_lay.addWidget(btn)

    def _failed(self, tb):
        QtWidgets.QMessageBox.critical(self, "Error", tb)
        self.stack.setCurrentWidget(self.landing)

class PhaseAWorker(QThread):
    progress = Signal(int, int, str)
    workers_snapshot = Signal(object)
    done = Signal(object)
    failed = Signal(str)
    def __init__(self, cfg, outdir): super().__init__(); self.cfg, self.outdir = cfg, outdir
    def run(self):
        try:
            cfg = self.cfg; mat = build_material_tables(cfg); cfg["_mat"] = mat
            geom, n_max, radii, turns = build_geometry(cfg)
            presets = [("low", cfg["T_amb_low"]), ("mid", cfg["T_amb_mid"]), ("hot", cfg["T_amb_hot"])]
            mon = WorkerMonitor(interval=1.0); mon.on_snapshot = self.workers_snapshot.emit
            cold_rows = run_cold_sweep(cfg, mat, geom, presets,
                                       progress=lambda d,t,m: self.progress.emit(d,t,m), monitor=mon)
            menu = post.write_phase_a(self.outdir, cfg, cold_rows, geom, radii, presets)
            self.done.emit(dict(mat=mat, geom=geom, radii=radii, presets=presets, cold_rows=cold_rows, menu=menu))
        except Exception: self.failed.emit(traceback.format_exc())

class PhaseCDWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)
    def __init__(self, cfg, study, choice, outdir): super().__init__(); self.cfg, self.study, self.choice, self.outdir = cfg, study, choice, outdir
    def run(self):
        try:
            R, n = self.choice["R"], self.choice["n"]; total = 2 * len(self.study["presets"]) + 1
            cold, cool, cool_rep = run_cooldowns(self.cfg, self.study["mat"], R, n, self.study["presets"], self.study["cold_rows"],
                                                 progress=lambda d,t,m: self.progress.emit(d,total,m))
            hot = run_hot(self.cfg, self.study["mat"], R, n, self.study["presets"], cool,
                          progress=lambda d,t,m: self.progress.emit(len(self.study["presets"])+d,total,m))
            res = post.write_final(self.outdir, self.cfg, R, n, self.study["cold_rows"], cold, cool_rep, hot, self.study["presets"])
            res.update(R=R, n=n, L=self.choice["L"])
            self.done.emit(res)
        except Exception: self.failed.emit(traceback.format_exc())

def main():
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion"); app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()