# Tank Coil Design Simulation (P6 engine, desktop app)

Finds the best heating coil for a tank of phase-change oil. It sweeps many coil designs, lets you pick one,
then tests that coil after the tank has cooled down and is reheated.

## Install and run
```
pip install -r requirements.txt
python main.py
```
Python 3.10 - 3.12 recommended (Numba must support your Python version).
The first start compiles the solver (a splash screen shows this; up to about a minute). Later starts reuse the shared cache.

## What you do
1. **Inputs** - fill the tabs (ambients, coil fluid, oil, wall and tube, agitator, tank size, windows, N, dt).
   Property tables have 5 temperature points; water and the P6 defaults are pre-filled and follow your
   temperature range until you edit a cell. Press **Review & start Phase A** and check the summary.
2. **Phase A - cold sweep** runs (progress bar, cancel button). When it finishes it writes its own output
   (`cold_low/mid/hot.png`, `picker_menu.txt`) before anything else happens.
3. **Phase B - pick a coil** in the table (liquid mass per design and ambient; best values highlighted).
   This is the only decision you make.
4. **Phases C and D** run automatically: cooldown with the coil's metal and water as stagnant thermal
   mass, then a hot run from the residual state.
5. The final files are written. B, C and D write nothing on their own.

## Output - `out/run_<timestamp>/`
| File | Written | Content |
|---|---|---|
| `cold_low.png`, `cold_mid.png`, `cold_hot.png` | after A | liquid / slurry / solid mass and temperature vs coil length, one curve per radius |
| `picker_menu.txt` | after A | the designs you choose from |
| `hot_start.png`, `selected_summary.png` | after D | hot-run and selected-coil summary graphs |
| `results.csv` | after D | 22 columns: every cold-sweep row, then the 3 hot-run rows (Windows line endings) |
| `index.html` | after D | six-section report; all five graphs are embedded in the page itself |

`out/last_inputs.json` remembers your last inputs. Use **Save / Load inputs...** for named input files.
"Pick a different coil" re-runs C and D on the same Phase A results and overwrites the final files of that run folder.

## Multi-core sweep (Phase A)
Phase A turns every simulation into its own task and runs the queue on a pool of worker **processes**
(each has its own interpreter, so the GIL is not a bottleneck). Results come back through `as_completed`,
so the progress bar is counted in one place with no locks. The solver is compiled once in the main process
and written to a shared on-disk cache (`%LOCALAPPDATA%\TankCoilSim\numba_cache` on Windows); the workers only
load it, which avoids the compile-time file-lock deadlock.
- Workers = all cores (max 8), one left free for the window on 4+ core machines. Set `VCH_WORKERS=N` to override
  (`VCH_WORKERS=1` runs everything in the app's own process).
- If worker processes cannot start, the sweep continues on threads automatically.
- Cancel stops the workers immediately.

## Model assumptions (all from P6)
- 1D radial enthalpy model; coil = heat source at radius R; whole turns n, pitch = tube OD, L = 2*pi*R*n.
- Radii: 0.33 to 0.75 of the tank radius (5 values); turns log-spaced from 1 to axial length / tube OD.
- The simulated oil mass is tank volume x density. A typed fluid mass is shown in the report only.
- Coarse N (< 50) makes liquid mass change in whole-cell steps; use N >= 100 for smooth curves.
- Cooldown starts from the selected coil's cold-run end state; the hot run starts from the cooldown result.

## Layout
```
main.py            window, background runner, picker, results view
vch/inputs.py      input form, defaults, validation, echo summary
vch/geometry.py    design grid, mesh, injection weights
vch/physics.py     enthalpy, oil/water properties, h_internal / h_external, eps-NTU
vch/cooldown.py    cooldown kernel + phase C
vch/sweep.py       heating kernel + phase A sweep + phase D
vch/post.py        graphs, picker menu, results.csv, index.html
```

## Packaging (not yet tested)
`pip install pyinstaller` then
`pyinstaller --noconfirm --windowed --name TankCoilSim --collect-all numba --collect-all llvmlite main.py`
