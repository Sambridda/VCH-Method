"""Resource monitoring tools."""
import os
import time
import threading
import queue as _queue
import psutil

class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.p = psutil.Process(os.getpid())
        self.interval = interval
        self.samples = []
        self._seen = {}
        self._stop = threading.Event()

    def _tree(self):
        procs = {self.p.pid: self.p}
        try:
            for ch in self.p.children(recursive=True):
                procs.setdefault(ch.pid, ch)
        except psutil.Error:
            pass
        self._seen.update(procs)
        return list(self._seen.values())

    def _loop(self):
        for p in self._tree():
            try: p.cpu_percent(None)
            except psutil.Error: pass
        while not self._stop.wait(self.interval):
            cpu = rss = 0.0
            for p in self._tree():
                try:
                    cpu += p.cpu_percent(None)
                    rss += p.memory_info().rss
                except psutil.Error: pass
            self.samples.append((time.perf_counter() - self.t0, cpu, rss))

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()
        self.wall = time.perf_counter() - self.t0
        cpu = [s[1] for s in self.samples]
        self.avg_cores = (sum(cpu) / len(cpu) / 100.0) if cpu else 0.0
        self.peak_rss_mb = max((s[2] for s in self.samples), default=0) / 1e6

class WorkerMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self._lock = threading.Lock()
        self._workers = {}
        self._slot = 0
        self._q = None
        self._thread = None
        self._stop = threading.Event()
        self.on_snapshot = None

    def start(self, status_q):
        self._q = status_q
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._drain()
            if self.on_snapshot:
                self.on_snapshot(self.snapshot())

    def _drain(self):
        while True:
            try: msg = self._q.get_nowait()
            except (_queue.Empty, OSError): return
            self._apply(msg)

    def _apply(self, m):
        with self._lock:
            pid = m["pid"]
            w = self._workers.get(pid)
            if w is None:
                self._slot += 1
                w = self._workers[pid] = dict(
                    pid=pid, slot=self._slot, state="starting", task=None,
                    done=0, cpu_busy=0.0, wall_busy=0.0,
                    first_wall=m["t_wall"], last_wall=m["t_wall"],
                    started_wall=None, started_cpu=None,
                    last_end_wall=None, idle_wall=0.0)
            w["last_wall"] = m["t_wall"]
            k = m["kind"]
            if k == "ready": w["state"] = "idle"
            elif k == "start":
                if w["last_end_wall"] is not None:
                    w["idle_wall"] += m["t_wall"] - w["last_end_wall"]
                w.update(state="running", task=m["idx"],
                         started_wall=m["t_wall"], started_cpu=m["t_cpu"])
            elif k == "end":
                if w["started_cpu"] is not None:
                    w["cpu_busy"] += m["t_cpu"] - w["started_cpu"]
                    w["wall_busy"] += m["t_wall"] - w["started_wall"]
                w.update(done=w["done"] + 1, task=None, state="idle",
                         last_end_wall=m["t_wall"])

    def snapshot(self):
        with self._lock:
            out = []
            for w in self._workers.values():
                alive = max(1e-6, w["last_wall"] - w["first_wall"])
                out.append({**w, "alive": alive, "utilization": w["cpu_busy"] / alive})
            return sorted(out, key=lambda x: x["slot"])