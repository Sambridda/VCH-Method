"""Tank coil design simulation - desktop application (P6 engine)."""
import io
import multiprocessing
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, QThread

from vch import inputs, post
from vch.physics import build_material_tables
from vch.geometry import build_geometry
from vch.sweep import run_cold_sweep, run_hot, run_heat, default_workers, Cancelled
from vch.cooldown import run_cooldowns, run_cool
from vch.resource_monitor import WorkerMonitor

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
OUT_ROOT = APP_DIR / "out"
LAST_INPUTS = OUT_ROOT / "last_inputs.json"

STYLE = """
QPushButton#primary { background:#1a4d7a; color:white; border:none; border-radius:4px; padding:6px 16px; font-weight:600; }
QPushButton#primary:hover { background:#256aa3; }
QPushButton#primary:disabled { background:#9fb3c6; }
QGroupBox { font-weight:600; margin-top:10px; }
QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }
QProgressBar { border:1px solid #bbb; border-radius:4px; text-align:center; height:22px; }
QProgressBar::chunk { background:#2f7fc1; }
"""

def warm_up_engine(app):
    splash = QtWidgets.QLabel("Preparing the numerical engine...\n\n"
                              "The first start on this computer compiles the solver and can take\n"
                              "up to a minute. Later starts are almost instant.")
    splash.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
    splash.setAlignment(Qt.AlignmentFlag.AlignCenter)
    splash.setStyleSheet("QLabel { background:#f4f7fa; border:2px solid #1a4d7a; padding:28px; font-size:13px; }")
    splash.show()
    app.processEvents()
    try:
        cfg, _errors, _warnings = inputs.build_cfg(inputs.default_inputs())
        mat = build_material_tables(cfg)
        R, n = 0.5 * cfg["r_inner"], 3
        r = run_heat(cfg, mat, R, n, cfg["T_amb_low"], None, 0.0)
        run_cool(cfg, mat, R, n, cfg["T_amb_low"], {"H": r["H_final"], "T_steel": r["T_steel"]}, 0.0)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(2, 2))
        fig.subplots().plot([0, 1])
        fig.tight_layout()
        fig.savefig(io.BytesIO(), format="png")
    finally:
        splash.close()

class PhaseAWorker(QThread):
    progress = Signal(int, int, str)
    workers_snapshot = Signal(object)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg, outdir):
        super().__init__()
        self.cfg, self.outdir = cfg, outdir
        self.cancel_event = threading.Event()

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
                                       cancel=self.cancel_event,
                                       monitor=mon)
            self.progress.emit(len(cold_rows), len(cold_rows), "Writing Phase A output (graphs + picker menu)...")
            menu = post.write_phase_a(self.outdir, cfg, cold_rows, geom, radii, presets)
            self.done.emit(dict(mat=mat, geom=geom, radii=radii, turns=turns, presets=presets,
                                cold_rows=cold_rows, menu=menu))
        except Cancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

class PhaseCDWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg, study, choice, outdir):
        super().__init__()
        self.cfg, self.study, self.choice, self.outdir = cfg, study, choice, outdir
        self.cancel_event = threading.Event()

    def run(self):
        try:
            cfg, st = self.cfg, self.study
            R, n = self.choice["R"], self.choice["n"]
            presets, mat = st["presets"], st["mat"]
            total = 2 * len(presets) + 1
            emit = self.progress.emit
            cold_sel, cool_states, cool_reports = run_cooldowns(
                cfg, mat, R, n, presets, st["cold_rows"],
                progress=lambda d, t, m: emit(d, total, "Phase C: " + m), cancel=self.cancel_event)
            hot_rows = run_hot(
                cfg, mat, R, n, presets, cool_states,
                progress=lambda d, t, m: emit(len(presets) + d, total, "Phase D: " + m),
                cancel=self.cancel_event)
            emit(total - 1, total, "Writing final output (results.csv, index.html)...")
            res = post.write_final(self.outdir, cfg, R, n, st["cold_rows"], cold_sel,
                                   cool_reports, hot_rows, presets)
            emit(total, total, "Done")
            res.update(R=R, n=n, L=self.choice["L"])
            self.done.emit(res)
        except Cancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())

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
            self.setPixmap(self._pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)

def image_tabs(names, folder):
    tabs = QtWidgets.QTabWidget()
    for name in names:
        img = ScaledImage()
        p = Path(folder) / name
        if p.exists():
            img.set_image(p)
        tabs.addTab(img, name.replace(".png", ""))
    return tabs

def open_path(path):
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

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
        self.worker_table.setHorizontalHeaderLabels(
            ["#", "PID", "State", "Task", "Done", "CPU s", "Utilization"])
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
            total_h = header_h + (len(snap) * row_h) + 4
            t.setFixedHeight(total_h)
            
        for r, w in enumerate(snap):
            vals = [str(w["slot"]), str(w["pid"]), w["state"],
                    str(w["task"]) if w["task"] is not None else "-",
                    str(w["done"]), f"{w['cpu_busy']:.1f}",
                    f"{100*w['utilization']:.0f}%"]
            
            for c, v in enumerate(vals):
                it = t.item(r, c)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    t.setItem(r, c, it)
                
                it.setText(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                it.setForeground(QtGui.QColor(20, 20, 20))
                
            color = {"running": QtGui.QColor(214, 238, 214),
                     "idle":    QtGui.QColor(255, 244, 200),
                     "starting": QtGui.QColor(230, 230, 230)}.get(w["state"])
            
            if color:
                for c in range(t.columnCount()):
                    t.item(r, c).setBackground(color)
                    
        busy = sum(w["utilization"] for w in snap)
        self.cores_label.setText(
            f"Cores busy (wall-average): {busy:.2f} / {os.cpu_count()} logical"
            f"   —   {len(snap)} workers")

class MainWindow(QtWidgets.QMainWindow):
    TAB_INPUTS, TAB_A, TAB_PICK, TAB_RESULTS = range(4)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tank Coil Design Simulation (P6)")
        self.resize(1180, 820)
        self.cfg = None
        self.study = None
        self.outdir = None
        self.worker = None
        self.menu = []

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.form = inputs.InputsForm()
        self.form.confirmed.connect(self.start_phase_a)
        self.tabs.addTab(self.form, "1  Inputs")

        self.page_a = ProgressPage()
        self.page_a.cancel_requested.connect(self.cancel_worker)
        self.tabs.addTab(self.page_a, "2  Phase A: cold sweep")

        self.tabs.addTab(self._build_pick_page(), "3  Phase B: pick coil")

        self.page_cd = ProgressPage()
        self.page_cd.cancel_requested.connect(self.cancel_worker)
        self.results_box = QtWidgets.QWidget()
        self.results_lay = QtWidgets.QVBoxLayout(self.results_box)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.page_cd)
        self.stack.addWidget(self.results_box)
        self.tabs.addTab(self.stack, "4  Phases C && D: results")

        for i in (self.TAB_A, self.TAB_PICK, self.TAB_RESULTS):
            self.tabs.setTabEnabled(i, False)
        self.statusBar().showMessage("Enter the inputs, then review and start Phase A.")

        if LAST_INPUTS.exists():
            try:
                self.form.set_inputs(inputs.load_inputs(str(LAST_INPUTS)))
                self.statusBar().showMessage("Restored your last inputs.")
            except Exception:
                pass

    def _build_pick_page(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        head = QtWidgets.QLabel("<b>Phase B - pick the coil to carry forward.</b> Each row is one design "
                                "(coil radius R, whole turns n); the liquid columns show how much oil is "
                                "liquid after the heating window from a cold start. "
                                "The best design in each column is highlighted.")
        head.setWordWrap(True)
        lay.addWidget(head)

        self.pick_table = QtWidgets.QTableWidget(0, 7)
        self.pick_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.pick_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.pick_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pick_table.verticalHeader().setVisible(False)
        self.pick_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pick_table.itemSelectionChanged.connect(self._pick_changed)
        self.pick_table.cellDoubleClicked.connect(lambda *_: self.start_phase_cd())

        self.pick_images = QtWidgets.QTabWidget()
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.pick_table)
        split.addWidget(self.pick_images)
        self.pick_table.setMinimumWidth(470)
        self.pick_images.setMinimumWidth(380)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        split.setSizes([520, 620])
        lay.addWidget(split, 1)

        bar = QtWidgets.QHBoxLayout()
        self.lbl_pick = QtWidgets.QLabel("Select a row.")
        self.btn_cd = QtWidgets.QPushButton("Run phases C && D with this coil  \u25B6")
        self.btn_cd.setObjectName("primary")
        self.btn_cd.setMinimumHeight(34)
        self.btn_cd.setEnabled(False)
        self.btn_cd.clicked.connect(self.start_phase_cd)
        self.btn_open_a = QtWidgets.QPushButton("Open picker_menu.txt")
        self.btn_open_a.clicked.connect(lambda: open_path(Path(self.outdir) / "picker_menu.txt"))
        self.btn_back = QtWidgets.QPushButton("\u25C0 Back to inputs")
        self.btn_back.clicked.connect(self.back_to_inputs)
        bar.addWidget(self.btn_back); bar.addWidget(self.btn_open_a)
        bar.addWidget(self.lbl_pick, 1)
        bar.addWidget(self.btn_cd)
        lay.addLayout(bar)
        return page

    def _fill_picker(self):
        cfg, menu, presets = self.cfg, self.menu, self.study["presets"]
        t = self.pick_table
        t.clearContents()
        t.setRowCount(len(menu))
        t.setHorizontalHeaderLabels(
            ["idx", "R (m)", "n", "L (m)"]
            + [f"liquid kg\n{p} ({T:g} \u00B0C)" for p, T in presets])
        best = {k: max(m[k] for m in menu) for k in ("liq_low", "liq_mid", "liq_hot")}
        for i, m in enumerate(menu):
            vals = [f"{m['idx']}", f"{m['R']:.3f}", f"{m['n']}", f"{m['L']:.1f}",
                    f"{m['liq_low']:.0f}", f"{m['liq_mid']:.0f}", f"{m['liq_hot']:.0f}"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                key = ("liq_low", "liq_mid", "liq_hot")[c - 4] if c >= 4 else None
                if key and m[key] >= best[key] - 1e-9 and best[key] > 0:
                    it.setBackground(QtGui.QColor(214, 238, 214))
                    it.setForeground(QtGui.QColor(20, 20, 20))
                    f = it.font(); f.setBold(True); it.setFont(f)
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

    def _chosen(self):
        rows = self.pick_table.selectionModel().selectedRows()
        return self.menu[rows[0].row()] if rows else None

    def start_phase_a(self, cfg, inp):
        try:
            inputs.save_inputs(str(LAST_INPUTS), inp)
        except OSError:
            pass
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = str(OUT_ROOT / f"run_{ts}")
        os.makedirs(self.outdir, exist_ok=True)

        self.tabs.setTabEnabled(self.TAB_INPUTS, False)
        self.tabs.setTabEnabled(self.TAB_PICK, False)
        self.tabs.setTabEnabled(self.TAB_RESULTS, False)
        self.tabs.setTabEnabled(self.TAB_A, True)
        self.tabs.setCurrentIndex(self.TAB_A)
        
        self.page_a.start("Phase A - cold-start sweep",
                          f"Output folder: {self.outdir}", show_workers=True)
        
        self.page_a.log_line(f"Using up to {default_workers()} worker process(es) "
                             f"({os.cpu_count()} CPU cores detected).")
        self.statusBar().showMessage("Phase A running...")

        self.worker = PhaseAWorker(cfg, self.outdir)
        self.worker.progress.connect(self.page_a.update_progress)
        self.worker.workers_snapshot.connect(self.page_a.update_workers)
        self.worker.done.connect(self._phase_a_done)
        self.worker.failed.connect(lambda tb: self._failed(tb, self.TAB_INPUTS))
        self.worker.cancelled.connect(lambda: self._cancelled(self.TAB_INPUTS))
        self.worker.start()

    def _phase_a_done(self, study):
        self.study = study
        self.menu = study["menu"]
        self.worker = None
        self._fill_picker()
        self.tabs.setTabEnabled(self.TAB_INPUTS, True)
        self.tabs.setTabEnabled(self.TAB_PICK, True)
        self.tabs.setCurrentIndex(self.TAB_PICK)
        self.page_a.btn_cancel.setEnabled(False)
        self.statusBar().showMessage(
            f"Phase A complete - cold graphs and picker_menu.txt written to {self.outdir}. Pick a coil.")

    def start_phase_cd(self):
        choice = self._chosen()
        if choice is None or self.worker is not None:
            return
        self.tabs.setTabEnabled(self.TAB_INPUTS, False)
        self.tabs.setTabEnabled(self.TAB_PICK, False)
        self.tabs.setTabEnabled(self.TAB_RESULTS, True)
        self.stack.setCurrentWidget(self.page_cd)
        self.tabs.setCurrentIndex(self.TAB_RESULTS)
        
        self.page_cd.start(
            f"Phases C & D - R = {choice['R']:.3f} m, n = {choice['n']}",
            f"Cooldown of {self.cfg['t_cooldown_hr']:.1f} h, then hot run of {self.cfg['t_window_hr']:.1f} h...",
            show_workers=False)
        
        self.statusBar().showMessage("Phases C and D running...")

        self.worker = PhaseCDWorker(self.cfg, self.study, choice, self.outdir)
        self.worker.progress.connect(self.page_cd.update_progress)
        self.worker.done.connect(self._phase_cd_done)
        self.worker.failed.connect(lambda tb: self._failed(tb, self.TAB_PICK))
        self.worker.cancelled.connect(lambda: self._cancelled(self.TAB_PICK))
        self.worker.start()

    def _phase_cd_done(self, res):
        self.worker = None
        self._build_results(res)
        self.stack.setCurrentWidget(self.results_box)
        self.tabs.setTabEnabled(self.TAB_INPUTS, True)
        self.tabs.setTabEnabled(self.TAB_PICK, True)
        self.statusBar().showMessage(f"Done - results.csv and index.html written to {self.outdir}")

    def _build_results(self, res):
        while self.results_lay.count():
            w = self.results_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        head = QtWidgets.QLabel(
            f"<b>Finished.</b> Selected coil: R = {res['R']:.3f} m, n = {res['n']}, L = {res['L']:.1f} m<br>"
            f"Files in <code>{self.outdir}</code>")
        head.setTextFormat(Qt.TextFormat.RichText)
        head.setWordWrap(True)
        self.results_lay.addWidget(head)

        text = QtWidgets.QPlainTextEdit(res["text"])
        text.setReadOnly(True)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(text)
        split.addWidget(image_tabs(post.PNG_NAMES, self.outdir))
        text.setMinimumWidth(430)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1)
        split.setSizes([500, 640])
        self.results_lay.addWidget(split, 1)

        bar = QtWidgets.QHBoxLayout()
        b_html = QtWidgets.QPushButton("Open report (index.html)")
        b_html.setObjectName("primary")
        b_html.clicked.connect(lambda: open_path(res["index_html"]))
        b_csv = QtWidgets.QPushButton("Open results.csv")
        b_csv.clicked.connect(lambda: open_path(res["results_csv"]))
        b_dir = QtWidgets.QPushButton("Open output folder")
        b_dir.clicked.connect(lambda: open_path(self.outdir))
        b_repick = QtWidgets.QPushButton("Pick a different coil")
        b_repick.clicked.connect(lambda: self.tabs.setCurrentIndex(self.TAB_PICK))
        b_new = QtWidgets.QPushButton("New study")
        b_new.clicked.connect(self.back_to_inputs)
        for b in (b_html, b_csv, b_dir):
            bar.addWidget(b)
        bar.addStretch()
        bar.addWidget(b_repick); bar.addWidget(b_new)
        self.results_lay.addLayout(bar)

    def back_to_inputs(self):
        self.tabs.setCurrentIndex(self.TAB_INPUTS)

    def cancel_worker(self):
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.statusBar().showMessage("Cancelling - finishing the runs already in progress...")

    def _cancelled(self, return_tab):
        self.worker = None
        self.tabs.setTabEnabled(self.TAB_INPUTS, True)
        self.tabs.setTabEnabled(self.TAB_PICK, return_tab == self.TAB_PICK)
        self.tabs.setCurrentIndex(return_tab)
        self.statusBar().showMessage("Cancelled.")

    def _failed(self, tb, return_tab):
        self.worker = None
        self.tabs.setTabEnabled(self.TAB_INPUTS, True)
        self.tabs.setTabEnabled(self.TAB_PICK, return_tab == self.TAB_PICK)
        self.tabs.setCurrentIndex(return_tab)
        self.statusBar().showMessage("The run failed.")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("The simulation stopped with an error")
        box.setText("Something went wrong while running the simulation.")
        box.setDetailedText(tb)
        box.exec()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.worker.wait(8000)
        event.accept()

def main():
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    warm_up_engine(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()